"""End-to-end integration test: one full inference + training step on mock data.

This is the test that matters most: it wires the whole stack together (mock
parser -> serializer -> mock semantic mapper -> mock memory -> tokenizer ->
feature builder -> model -> losses) and verifies a single forward/backward step
runs and produces finite, correctly-shaped outputs.
"""

import torch

from nsm_ct import build_default_stack, load_config
from nsm_ct.data_structures import ComprehensionExample
from nsm_ct.dataset import ComprehensionDataset, generate_toy_dataset, make_dataloader
from nsm_ct.features import FeatureBuilder
from nsm_ct.losses import compute_losses
from nsm_ct.model import ConsciousnessTransformer
from nsm_ct.parser_interface import MockNaomiParser
from nsm_ct.serialization import serialize_parse_tree


def _small_config():
    cfg = load_config()
    # shrink for a fast test
    cfg.model.d_model = 32
    cfg.model.num_layers = 1
    cfg.model.nhead = 2
    cfg.model.dim_feedforward = 32
    cfg.data.num_examples = 8
    cfg.train.batch_size = 4
    return cfg


def test_parser_and_serialization_pathway():
    parser = MockNaomiParser()
    tree = parser.parse("The cat is red.")
    tokens = serialize_parse_tree(tree)
    assert tokens[0] == "[TREE]" and tokens[-1] == "[/TREE]"
    # surface tokens survive serialization
    assert "cat" in tokens and "red" in tokens


def test_feature_builder_produces_four_options():
    cfg = _small_config()
    ex = ComprehensionExample("The cat is red.", "What color?", ["red", "blue", "green", "gray"], 0)
    tokenizer, fb = build_default_stack(cfg, [ex])
    assert isinstance(fb, FeatureBuilder)
    encoded = fb.build(ex)
    assert len(encoded.rows) == 4
    assert encoded.consciousness.shape == (cfg.model.consciousness_dim,)
    assert encoded.memory.shape == (cfg.model.memory_dim,)
    # the answer region is non-empty for each option
    assert all(sum(r.answer_mask) > 0 for r in encoded.rows)


def test_one_full_inference_and_training_step():
    cfg = _small_config()
    torch.manual_seed(0)

    examples = generate_toy_dataset(cfg.data.num_examples, seed=0)
    tokenizer, fb = build_default_stack(cfg, examples)
    ds = ComprehensionDataset(examples, fb)
    loader = make_dataloader(ds, pad_id=tokenizer.pad_id, batch_size=cfg.train.batch_size, shuffle=False)

    model = ConsciousnessTransformer(tokenizer.vocab_size, cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = next(iter(loader))
    b = batch.answer_idx.shape[0]

    # --- forward (inference) ---
    output = model(batch)
    assert output.lm_logits.shape[0] == b * batch.num_options
    assert output.lm_logits.shape[-1] == tokenizer.vocab_size
    assert output.option_logits.shape == (b, batch.num_options)
    assert output.next_consciousness.shape == (b * batch.num_options, cfg.model.consciousness_dim)

    preds = model.predict(batch)
    assert preds.shape == (b,)
    assert preds.min() >= 0 and preds.max() < batch.num_options

    # --- loss + backward (one training step) ---
    losses = compute_losses(output, batch, weight_lm=1.0, weight_answer=1.0, weight_consistency=0.1)
    assert torch.isfinite(losses.total)
    for component in (losses.lm, losses.answer, losses.consistency):
        assert torch.isfinite(component)

    optimizer.zero_grad()
    losses.total.backward()
    # gradients actually flow to model parameters
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()


def test_loss_decreases_when_overfitting_one_batch():
    """Sanity check that training actually optimizes the objective."""
    cfg = _small_config()
    torch.manual_seed(0)
    examples = generate_toy_dataset(8, seed=0)
    tokenizer, fb = build_default_stack(cfg, examples)
    ds = ComprehensionDataset(examples, fb)
    loader = make_dataloader(ds, pad_id=tokenizer.pad_id, batch_size=8, shuffle=False)
    batch = next(iter(loader))

    model = ConsciousnessTransformer(tokenizer.vocab_size, cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    first = None
    last = None
    for step in range(15):
        output = model(batch)
        losses = compute_losses(output, batch, 1.0, 1.0, 0.1)
        optimizer.zero_grad()
        losses.total.backward()
        optimizer.step()
        if step == 0:
            first = float(losses.total.detach())
        last = float(losses.total.detach())

    assert last < first  # the model learned *something* on this batch
