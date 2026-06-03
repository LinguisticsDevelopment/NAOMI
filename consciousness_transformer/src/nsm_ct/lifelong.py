"""Lifelong loop: learn more and more over successive tests.

Runs many rounds of episodes ("tests"). Each round does two complementary kinds
of learning:

* **parametric** — a normal training step updates the model weights, and
* **non-parametric** — the local context absorbed during each episode is
  *consolidated* into the persistent :class:`~nsm_ct.long_term_memory.LongTermMemory`,
  growing a repository of entries and connections that future rounds can read
  from.

Over rounds you should see answer accuracy improve (weights learning) *and* the
long-term repo grow (knowledge accumulating). The repo can be saved to disk so it
persists across sessions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from . import Stack, build_default_stack
from .config import Config
from .dataset import EpisodeDataset, make_dataloader
from .episode import make_source
from .losses import compute_losses
from .metrics import action_accuracy, answer_accuracy


def run_lifelong(
    config: Config,
    num_rounds: int = 10,
    episodes_per_round: int = 32,
    lr: Optional[float] = None,
    ltm_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[Stack, List[dict]]:
    """Run the lifelong loop and return ``(stack, history)``.

    Args:
        config: Base config; long-term memory is force-enabled.
        num_rounds: How many rounds ("tests") of fresh episodes to run.
        episodes_per_round: New episodes generated each round (input→response
            cycles).
        lr: Optional learning-rate override.
        ltm_path: If set, save the long-term repo here at the end.
        verbose: Print per-round progress.

    Returns:
        The wired :class:`Stack` and a per-round metrics history.
    """
    config.model.use_long_term = True
    torch.manual_seed(config.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source = make_source(
        config.data.source, seed=config.data.seed, max_level=config.data.max_level,
        babi_task=config.data.babi_task, babi_path=config.data.babi_path,
    )
    # A pool only to build a stable tokenizer/vocab; the lexicon is fixed so new
    # rounds stay in-vocab.
    pool = source.generate(max(config.data.num_episodes, episodes_per_round * 2))
    stack = build_default_stack(config, pool)
    stack.mind.to(device)
    optimizer = torch.optim.AdamW(stack.mind.parameters(), lr=lr or config.train.learning_rate)

    history: List[dict] = []
    for r in range(num_rounds):
        episodes = source.generate(episodes_per_round)  # fresh "material"
        ds = EpisodeDataset(episodes, stack.encoder, stack.tokenizer, stack.answer_vocab, config)
        loader = make_dataloader(ds, stack.tokenizer.pad_id, config.train.batch_size, shuffle=True)

        stack.mind.train()
        loss_sum = acc_sum = act_sum = 0.0
        added = nb = 0
        for batch in loader:
            batch = batch.to(device)
            out = stack.mind(batch)
            losses = compute_losses(
                out, batch,
                weight_answer=config.train.weight_answer,
                weight_action=config.train.weight_action,
                weight_consistency=config.train.weight_consistency,
            )
            optimizer.zero_grad()
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(stack.mind.parameters(), config.train.grad_clip)
            optimizer.step()
            # Non-parametric: grow the long-term repo (outside the gradient).
            added += stack.mind.consolidate(out)
            loss_sum += float(losses.total.detach())
            acc_sum += answer_accuracy(out, batch)
            act_sum += action_accuracy(out, batch)
            nb += 1

        stats = stack.long_term.stats()
        rec = {
            "round": r + 1,
            "loss": loss_sum / max(nb, 1),
            "ans_acc": acc_sum / max(nb, 1),
            "act_acc": act_sum / max(nb, 1),
            "added": added,
            "ltm_entries": stats["entries"],
            "ltm_connections": stats["connections"],
        }
        history.append(rec)
        if verbose:
            print(
                f"round {rec['round']:>3}/{num_rounds} | loss={rec['loss']:.4f} "
                f"ans_acc={rec['ans_acc']:.3f} act_acc={rec['act_acc']:.3f} "
                f"| +{rec['added']} -> LTM entries={rec['ltm_entries']} "
                f"connections={rec['ltm_connections']}"
            )

    if ltm_path:
        stack.long_term.save(ltm_path)
        if verbose:
            print(f"Saved long-term memory to {ltm_path}")

    return stack, history
