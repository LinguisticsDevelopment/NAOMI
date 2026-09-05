"""The TRAINED reconstruction decoder (learned realizer).

RESEARCH_NOTES.md "DECODER PLAN UPDATE" (lead, 2026-09-05) supersedes the
Phase-1 deterministic-only plan in `dev/DECODER_DESIGN.md`: the decoder is now
a small LEARNED model, trained SELF-SUPERVISED by RECONSTRUCTION and tested by
ROUND-TRIP (autoencoder):

    text -> ENCODER -> grounded structure -> DECODER -> text ,  output == input

The source sentence IS its own label -- no separate decoder gold is needed.
Training data is `encoder_gold_v2.jsonl`: each record's TOP tree
(`lattice.trees[0]`) is the committed structure, and the record's own `text`
tokens are the reconstruction target. Reconstruction needs the tree's
STRUCTURE (clauses/roles + which surface token realized each node) and NOT
sense disambiguation, so this module never reads `sense_id`/candidate lists
-- it reads exactly `(relation, grounding.type, token_index)` per node, via
`encoder_model.clause_node_order` (the same walk the encoder's own oracle
uses), and the node's realized surface word straight from `record["tokens"]`.

NO-CONFAB, enforced by construction (the same discipline as `decoder.py`'s
Phase-1 gate, `dev/DECODER_DESIGN.md` §4):
  - The decoder never has a free vocabulary distribution. Every step's output
    is one of (a) COPY one of the structure's own nodes' surface words, or
    (b) GENERATE from a small CLOSED function-word vocabulary mined from the
    gap positions the structure never covers (articles, copulas-as-glue,
    punctuation, ...) -- built once from the training corpus, never learned
    as a free softmax over an open vocabulary.
  - `realize()` gates on structure content BEFORE ever invoking the decoder
    network: if the handed structure carries no node with a real word (the
    ablation's severed case), there is nothing to copy and nothing to say --
    it returns `[]` immediately, exactly as Phase-1 abstains when a grounding
    is missing. A trained decoder that could still emit fluent content with
    every content node nulled would have learned knowledge in its weights;
    this module makes that path structurally absent, not just untrained-away.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import encoder_model as em

BOS = "<bos>"
UNK_FUNC = "<unk>"


# ---------------------------------------------------------------------------
# The committed structure (one tree, per RESEARCH_NOTES: "for training data
# use each encoder_gold_v2 record's TOP tree ... sense DISAMBIGUATION is not
# needed to regenerate surface -- that's comprehension's job")
# ---------------------------------------------------------------------------

@dataclass
class Node:
    token_index: int
    word: str
    relation: str
    gtype: str


@dataclass
class CommittedStructure:
    nodes: List[Node] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)   # reconstruction target, when known (training/eval)


def extract_nodes(record: dict, tree: dict) -> List[Node]:
    """A tree's (predicate + role) nodes, in `clause_node_order`'s canonical
    left-to-right walk (the same one the encoder oracle linearizes) -- reused
    here rather than re-derived, so encoder and decoder agree on what a
    "node" is. Only nodes with a real surface `token_index` are kept: a
    synthesized/elided filler (prime `YOU`, an elided predicate) has no
    literal token in the source text, so it is not a reconstruction target
    and carries nothing to copy.
    """
    nodes: List[Node] = []
    for clause in tree.get("clauses", []):
        for relation, grounding, tidx in em.clause_node_order(record, clause):
            if tidx is None:
                continue
            nodes.append(Node(token_index=tidx, word=record["tokens"][tidx],
                               relation=relation, gtype=grounding.get("type")))
    return nodes


def build_structure(record: dict, tree: dict) -> CommittedStructure:
    return CommittedStructure(nodes=extract_nodes(record, tree), tokens=list(record["tokens"]))


def sever_structure_content(structure: CommittedStructure) -> CommittedStructure:
    """The no-confab ablation (design §4.2's construction, applied to the
    learned decoder): null every node's surface word while keeping the
    structure's shape (token positions/relations/types) -- the structure a
    comprehension/encoder path would hand off severed memory. `realize()`
    must collapse to empty on this input; see the module docstring.
    """
    severed = [Node(token_index=n.token_index, word=None, relation=n.relation, gtype=n.gtype)
               for n in structure.nodes]
    return CommittedStructure(nodes=severed, tokens=list(structure.tokens))


# ---------------------------------------------------------------------------
# The closed function-word vocabulary (the ONLY generation the decoder can
# do; everything else is a copy from the handed structure)
# ---------------------------------------------------------------------------

def build_function_vocab(records: Sequence[dict]) -> List[str]:
    """The lowercased surface tokens that fall in the GAPS a tree's own nodes
    never cover (articles, copulas realized elsewhere, punctuation, stray
    duplicate tokens, ...) -- mined once, deterministically, from the top
    tree of every training record. `<unk>` is always slot 0, the closed
    vocabulary's own fallback for a gap word never seen in training.
    """
    gaps = set()
    for r in records:
        trees = r.get("lattice", {}).get("trees") or []
        if not trees:
            continue
        covered = {n.token_index for n in extract_nodes(r, trees[0])}
        for i, tok in enumerate(r["tokens"]):
            if i not in covered:
                gaps.add(tok.lower())
    return [UNK_FUNC] + sorted(gaps - {UNK_FUNC})


# ---------------------------------------------------------------------------
# Tensor features (b): a committed structure -> encode-side tensors, and (for
# training) the reconstruction target as an EXTENDED-VOCABULARY label per
# step: index < M is "copy node M", index >= M is "generate function-vocab[
# index - M]" (the last such index is reserved for EOS). No index is ever a
# free open-vocabulary id.
# ---------------------------------------------------------------------------

@dataclass
class DecoderFeatures:
    node_word_hash: torch.Tensor      # LongTensor[M]
    node_relation_id: torch.Tensor    # LongTensor[M]
    node_gtype_id: torch.Tensor       # LongTensor[M]
    node_words: List[str]             # length M -- the copy source text
    target_tokens: List[str]          # length T -- record["tokens"]
    target_labels: torch.Tensor       # LongTensor[T+1] (T reconstruction steps + EOS)
    prev_token_hash: torch.Tensor     # LongTensor[T+1] teacher-forced previous-token input


def _node_tensors(nodes: Sequence[Node], relation_vocab: Dict[str, int], hash_buckets: int):
    if not nodes:
        z = torch.zeros(0, dtype=torch.long)
        return z, z.clone(), z.clone()
    word_hash = torch.tensor([em.hash_bucket(n.word.lower(), hash_buckets) for n in nodes], dtype=torch.long)
    relation_id = torch.tensor([relation_vocab.get(n.relation, relation_vocab[em.UNK]) for n in nodes],
                                dtype=torch.long)
    gtype_id = torch.tensor([em.GTYPE_INDEX.get(n.gtype, len(em.GROUNDING_TYPES)) for n in nodes],
                             dtype=torch.long)
    return word_hash, relation_id, gtype_id


def build_decoder_features(record: dict, tree: dict, function_vocab: Sequence[str],
                            relation_vocab: Dict[str, int], hash_buckets: int = 2048) -> DecoderFeatures:
    structure = build_structure(record, tree)
    nodes = [n for n in structure.nodes if n.word is not None]
    word_hash, relation_id, gtype_id = _node_tensors(nodes, relation_vocab, hash_buckets)

    func_index = {w: i for i, w in enumerate(function_vocab)}
    unk_id = func_index[UNK_FUNC]
    node_by_tidx: Dict[int, int] = {}
    for idx, n in enumerate(nodes):
        node_by_tidx.setdefault(n.token_index, idx)

    M = len(nodes)
    tokens = structure.tokens
    eos_label = M + len(function_vocab)   # one past the last function-vocab id

    labels: List[int] = []
    prev_tokens = [BOS]
    for t, tok in enumerate(tokens):
        if t in node_by_tidx:
            labels.append(node_by_tidx[t])
        else:
            labels.append(M + func_index.get(tok.lower(), unk_id))
        prev_tokens.append(tok)
    labels.append(eos_label)

    prev_hash = torch.tensor([em.hash_bucket(w.lower(), hash_buckets) for w in prev_tokens], dtype=torch.long)
    return DecoderFeatures(
        node_word_hash=word_hash, node_relation_id=relation_id, node_gtype_id=gtype_id,
        node_words=[n.word for n in nodes], target_tokens=list(tokens),
        target_labels=torch.tensor(labels, dtype=torch.long), prev_token_hash=prev_hash,
    )


# ---------------------------------------------------------------------------
# The model (a): a small GRU encoder over the linearized structure + a
# GRU-cell decoder with dot-product attention/copy over the structure's own
# node encodings, extended with a closed function-word generation head.
# SUB-MB by construction (see the module test / train_decoder.py's printed
# parameter count).
# ---------------------------------------------------------------------------

class DecoderTrainedModel(nn.Module):
    def __init__(self, relation_vocab: Dict[str, int], function_vocab: Sequence[str],
                 hash_buckets: int = 2048, d_tok: int = 24, d_rel: int = 12, d_type: int = 6,
                 d_model: int = 48):
        super().__init__()
        self.relation_vocab = relation_vocab
        self.function_vocab = list(function_vocab)
        self.hash_buckets = hash_buckets
        self.d_model = d_model
        self.func_vocab_size = len(self.function_vocab)
        self.eos_local_id = self.func_vocab_size   # index into func_head's output, one past the words

        self.tok_emb = nn.Embedding(hash_buckets, d_tok)
        self.rel_emb = nn.Embedding(len(relation_vocab), d_rel)
        self.type_emb = nn.Embedding(len(em.GROUNDING_TYPES) + 1, d_type)
        self.node_proj = nn.Linear(d_tok + d_rel + d_type, d_model)
        self.node_rnn = nn.GRU(d_model, d_model, batch_first=True)

        self.dec_rnn = nn.GRUCell(d_tok + d_model, d_model)
        self.func_head = nn.Linear(d_model, self.func_vocab_size + 1)   # + EOS

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode_structure(self, node_word_hash: torch.Tensor, node_relation_id: torch.Tensor,
                          node_gtype_id: torch.Tensor) -> torch.Tensor:
        """The structure's own nodes -> contextualized encodings [M, d_model].
        M may be 0 (an empty / fully-severed structure): the copy source is
        then empty and the decoder can only reach the function-word head.
        """
        if node_word_hash.numel() == 0:
            return torch.zeros(0, self.d_model)
        tok = self.tok_emb(node_word_hash)
        rel = self.rel_emb(node_relation_id)
        typ = self.type_emb(node_gtype_id)
        x = self.node_proj(torch.cat([tok, rel, typ], dim=-1)).unsqueeze(0)
        enc, _ = self.node_rnn(x)
        return enc.squeeze(0)

    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(1, self.d_model)

    def decode_step(self, prev_token_hash: torch.Tensor, h_prev: torch.Tensor,
                     node_enc: torch.Tensor):
        """One teacher-forced / greedy step. Returns (combined_logits[M+V+1],
        new_hidden). `combined_logits[:M]` are copy (pointer) scores over the
        structure's own nodes; `combined_logits[M:]` are the closed
        function-vocab + EOS logits. There is no third region -- the softmax
        over `combined_logits` can only ever select a copy or a closed-vocab
        generation, never an arbitrary vocabulary id.
        """
        prev_emb = self.tok_emb(prev_token_hash.view(1))            # [1, d_tok]
        M = node_enc.shape[0]
        if M > 0:
            scores = node_enc @ h_prev.squeeze(0)                    # [M]
            weights = torch.softmax(scores, dim=0)
            context = (weights.unsqueeze(-1) * node_enc).sum(dim=0, keepdim=True)  # [1, d_model]
        else:
            scores = torch.zeros(0)
            context = torch.zeros(1, self.d_model)
        h = self.dec_rnn(torch.cat([prev_emb, context], dim=-1), h_prev)
        func_logits = self.func_head(h).squeeze(0)                   # [V+1]
        combined = torch.cat([scores, func_logits], dim=0)           # [M+V+1]
        return combined, h


# ---------------------------------------------------------------------------
# (c) the reconstruction loss: teacher-forced next-token CE over the target
# surface tokens, conditioned on the structure.
# ---------------------------------------------------------------------------

def reconstruction_loss(model: DecoderTrainedModel, feats: DecoderFeatures) -> torch.Tensor:
    node_enc = model.encode_structure(feats.node_word_hash, feats.node_relation_id, feats.node_gtype_id)
    h = model.init_hidden()
    losses = []
    for t in range(feats.target_labels.shape[0]):
        combined, h = model.decode_step(feats.prev_token_hash[t], h, node_enc)
        losses.append(F.cross_entropy(combined.unsqueeze(0), feats.target_labels[t].view(1)))
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# (d) realize(): greedy decode, gated by the no-confab check BEFORE the
# network ever runs.
# ---------------------------------------------------------------------------

def realize(model: DecoderTrainedModel, structure: CommittedStructure, max_len: int = 40) -> List[str]:
    """A committed structure -> its reconstructed surface tokens.

    No-confab gate (design's ablation, applied to the learned decoder): if
    the structure carries no node with a real word, there is nothing to copy
    and nothing derivable from the closed vocabulary is grounded content --
    `realize` returns `[]` without ever invoking `dec_rnn`/`func_head`. This
    is a structural short-circuit, not a trained behaviour: the severed case
    can never reach the network at all.
    """
    nodes = [n for n in structure.nodes if n.word]
    if not nodes:
        return []

    word_hash, relation_id, gtype_id = _node_tensors(nodes, model.relation_vocab, model.hash_buckets)
    node_enc = model.encode_structure(word_hash, relation_id, gtype_id)
    h = model.init_hidden()
    prev_hash = torch.tensor(em.hash_bucket(BOS, model.hash_buckets))

    out: List[str] = []
    M = node_enc.shape[0]
    with torch.no_grad():
        for _ in range(max_len):
            combined, h = model.decode_step(prev_hash, h, node_enc)
            idx = int(torch.argmax(combined).item())
            if idx < M:
                word = nodes[idx].word
            else:
                local = idx - M
                if local == model.eos_local_id:
                    break
                word = model.function_vocab[local]
            out.append(word)
            prev_hash = torch.tensor(em.hash_bucket(word.lower(), model.hash_buckets))
    return out


# ---------------------------------------------------------------------------
# (e) round_trip + reconstruction_accuracy
# ---------------------------------------------------------------------------

def round_trip(record: dict, tree: dict, model: DecoderTrainedModel) -> str:
    """encode (gold structure, standing in for a real encoder+comprehension
    pass) -> DECODER -> text, RESEARCH_NOTES's round-trip acceptance test."""
    structure = build_structure(record, tree)
    return " ".join(realize(model, structure))


def reconstruction_accuracy(pred: Union[str, Sequence[str]],
                             gold: Union[str, Sequence[str]]) -> Dict[str, float]:
    """exact_match (whole-sequence, case-insensitive) + token-level F1
    (multiset overlap, case-insensitive) -- 100% exact is not required
    (design's own caveat: articles/inflection may vary)."""
    pred_toks = [t.lower() for t in (pred.split() if isinstance(pred, str) else pred)]
    gold_toks = [t.lower() for t in (gold.split() if isinstance(gold, str) else gold)]

    exact_match = 1.0 if pred_toks == gold_toks else 0.0

    common = Counter(pred_toks) & Counter(gold_toks)
    n_common = sum(common.values())
    if n_common == 0 or not pred_toks or not gold_toks:
        token_f1 = 0.0
    else:
        precision = n_common / len(pred_toks)
        recall = n_common / len(gold_toks)
        token_f1 = 2 * precision * recall / (precision + recall)

    return {"exact_match": exact_match, "token_f1": token_f1}


__all__ = [
    "Node", "CommittedStructure", "extract_nodes", "build_structure", "sever_structure_content",
    "build_function_vocab", "DecoderFeatures", "build_decoder_features",
    "DecoderTrainedModel", "reconstruction_loss", "realize", "round_trip", "reconstruction_accuracy",
]
