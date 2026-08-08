"""USVS — the Universal Semantic Vector Space (M29 / Step C).

The versioned artifact that combines the measured winners of M17–M28 into one
loadable, deterministic space:

- **Named axes** (primes + attr:* + lex:* + the M28.1 domain features) — every
  axis keeps its name; nothing is an opaque dimension.
- **The placed word core**: the gloss-vocabulary words placed by deterministic
  propagation (M19.2) over the winning close-edge set (synonym + similar +
  also_see — M28.1). This is the antonym-capable, relation-derived layer.
- **The full sense layer**: every WordNet synset grounded from its OWN gloss
  (the M22 per-sense grounding) as a sparse (axis, value) signature — the
  coordinate the WSD layer consumes, with total coverage.
- **The relational store** (structure the coordinate cannot carry — the M17–M25
  boundary finding): signed antonym edges with provenance tiers
  (direct | satellite_head | satellite_satellite — M28.1's expansion, tiered
  instead of naively filtered), and **directed** genus edges (M28.1's lesson:
  hypernymy is directional structure, not closeness).

Build discipline: the artifact is a PRODUCTION build — placement propagates over
ALL edges of the winning signal set (no held-out split). Quality numbers are
never read off this build; they come from the held-out harness
(`ground/ablation.py`, M24 rule). Rebuilds are deterministic: same code + same
WordNet → byte-identical arrays and the same fingerprint.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..nsm_primes import PRIME_NAMES
from .ablation import with_features
from .axes import MeaningAxes
from .cache import DecompCache
from .canonicalization import canon_label
from .corpus import gloss_vocabulary
from .definition_graph import content_words, naive_decompose
from .meaning_value import cosine
from .placement import place
from .relations import RelationGraph
from . import signal_also_see, signal_domain, signal_genus

SCHEMA_VERSION = "usvs-1"
_PRIME_SET = frozenset(PRIME_NAMES)

# Antonym provenance tiers, strongest first. Consumers choose their floor;
# the default query tier set excludes satellite_satellite (where M28.1's
# sampled artifact classes — compass chains, off-sense pairs — concentrate).
ANTONYM_TIERS = ("direct", "satellite_head", "satellite_satellite")
DEFAULT_ANTONYM_TIERS = ("direct", "satellite_head")


# ---------------------------------------------------------------------------
# Sense grounding (M22 semantics, word-level cache so 117k glosses stay cheap)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=200_000)
def _word_prime_counts(word: str, depth: int) -> Tuple[Tuple[str, int], ...]:
    counts: Dict[str, int] = {}
    for n in naive_decompose(word, max_depth=depth).iter_preorder():
        lab = canon_label(n.label)
        if lab in _PRIME_SET:
            counts[lab] = counts.get(lab, 0) + 1
    return tuple(sorted(counts.items()))


def sense_prime_weights(gloss: str, *, depth: int = 2, max_children: int = 8) -> Dict[str, float]:
    """`sense_graph.gloss_prime_weights` semantics with a word-level decompose
    cache (identical output — counts are additive per content word)."""
    counts: Dict[str, float] = {}
    for c in content_words(gloss or "")[:max_children]:
        for prime, n in _word_prime_counts(c, depth):
            counts[prime] = counts.get(prime, 0.0) + n
    if not counts:
        return {}
    m = max(counts.values())
    return {k: v / m for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Tiered antonym edges (M28.1 satellite expansion, with provenance kept)
# ---------------------------------------------------------------------------
def antonym_edges_tiered(words: Sequence[str]) -> List[Tuple[str, str, str]]:
    """(a, b, tier) for in-vocab pairs; each pair keeps its STRONGEST tier."""
    from ..wordnet import _wn  # module accessor; synset-level walk needs nltk
    wn = _wn()
    wset = set(words)
    best: Dict[Tuple[str, str], int] = {}

    def note(a: str, b: str, tier: int) -> None:
        a, b = a.lower(), b.lower()
        if a in wset and b in wset and a != b:
            key = tuple(sorted((a, b)))
            if tier < best.get(key, 99):
                best[key] = tier

    for w in words:
        for syn in wn.synsets(w):
            for lem in syn.lemmas():
                if lem.name().lower() != w:
                    continue
                for ant in lem.antonyms():                      # direct
                    note(w, ant.name(), 0)
            if syn.pos() not in ("a", "s"):
                continue
            heads = syn.similar_tos() if syn.pos() == "s" else [syn]
            for head in heads:
                for hl in head.lemmas():
                    for ant in hl.antonyms():                   # head's antonym
                        tier = 1 if syn.pos() == "s" else 0
                        note(w, ant.name(), tier)
                        for sat in ant.synset().similar_tos():  # its satellites
                            for sl in sat.lemmas():
                                note(w, sl.name(), 2)
    return sorted((a, b, ANTONYM_TIERS[t]) for (a, b), t in best.items())


# ---------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------
@dataclass
class USVS:
    version: str
    fingerprint: str
    axes: List[str]
    core_words: List[str]
    core_coords: np.ndarray                    # [W, D] float32, placed
    sense_ids: List[str]                       # sorted synset names
    sense_lemmas: List[List[str]]
    sense_indptr: np.ndarray                   # CSR over senses
    sense_axis_idx: np.ndarray
    sense_axis_val: np.ndarray                 # float32
    antonyms: List[Tuple[str, str, str]]       # (a, b, tier)
    genus: List[Tuple[str, str]]               # directed: word -> genus
    meta: Dict

    # -- queries -----------------------------------------------------------
    def word_coord(self, word: str) -> Optional[np.ndarray]:
        i = self._word_index.get(word.lower())
        return self.core_coords[i] if i is not None else None

    def sense_signature(self, sense_id: str) -> Dict[str, float]:
        i = self._sense_index.get(sense_id)
        if i is None:
            return {}
        lo, hi = self.sense_indptr[i], self.sense_indptr[i + 1]
        return {self.axes[j]: float(v)
                for j, v in zip(self.sense_axis_idx[lo:hi], self.sense_axis_val[lo:hi])}

    def sense_dense(self, sense_id: str) -> Optional[np.ndarray]:
        i = self._sense_index.get(sense_id)
        if i is None:
            return None
        v = np.zeros(len(self.axes), dtype=np.float32)
        lo, hi = self.sense_indptr[i], self.sense_indptr[i + 1]
        v[self.sense_axis_idx[lo:hi]] = self.sense_axis_val[lo:hi]
        return v

    def senses_of(self, word: str) -> List[str]:
        return self._lemma_index.get(word.lower(), [])

    def similarity(self, a: str, b: str) -> float:
        """Word-level: placed-core cosine when both are core words; otherwise
        MFS sense-signature cosine (WordNet orders senses MFS-first)."""
        ca, cb = self.word_coord(a), self.word_coord(b)
        if ca is not None and cb is not None:
            return cosine(ca, cb)
        sa, sb = self.senses_of(a), self.senses_of(b)
        if not sa or not sb:
            return 0.0
        return cosine(self.sense_dense(sa[0]), self.sense_dense(sb[0]))

    def antonyms_of(self, word: str,
                    tiers: Sequence[str] = DEFAULT_ANTONYM_TIERS) -> List[str]:
        w, ts = word.lower(), set(tiers)
        out = {b if a == w else a for a, b, t in self.antonyms
               if t in ts and w in (a, b)}
        return sorted(out)

    def genus_of(self, word: str) -> List[str]:
        w = word.lower()
        return sorted({g for a, g in self.genus if a == w})

    def __post_init__(self) -> None:
        self._word_index = {w: i for i, w in enumerate(self.core_words)}
        self._sense_index = {s: i for i, s in enumerate(self.sense_ids)}
        self._lemma_index: Dict[str, List[str]] = {}
        for sid, lemmas in zip(self.sense_ids, self.sense_lemmas):
            for lem in lemmas:
                self._lemma_index.setdefault(lem.lower(), []).append(sid)
        # WordNet sense order (MFS first) per lemma, not sorted-synset order
        from ..wordnet import senses as _senses
        for lem, sids in self._lemma_index.items():
            if len(sids) > 1:
                order = {s["sense_id"]: k for k, s in enumerate(_senses(lem))}
                sids.sort(key=lambda s: order.get(s, 999))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_usvs(*, n_core: int = 10_000, max_senses: Optional[int] = None,
               depth: int = 3, sense_depth: int = 2, alpha: float = 0.7,
               iters: int = 20, log=None) -> USVS:
    def say(msg: str) -> None:
        if log:
            log(msg)

    # 1) the word core over the winning signal set
    vocab = list(gloss_vocabulary(n_core))
    g0 = RelationGraph.build(vocab)
    g = with_features(g0, signal_domain.extras(vocab, g0)["feature_extra"])
    axes = MeaningAxes.assemble(g, min_attribute_freq=2)
    cache = DecompCache(depth=depth).warm(vocab)
    say(f"core: |vocab|={len(g.words())} axes={axes.dim}")

    words = g.words()
    wset = set(words)
    syn = sorted({tuple(sorted((w, s))) for w in words
                  for s in g.synonym.get(w, []) if s in wset and s != w})
    sim = sorted({tuple(sorted((w, s))) for w in words
                  for s in g.similar.get(w, []) if s in wset and s != w})
    also = signal_also_see.extras(vocab, g)["close_extra"]
    placed = place(words, g, axes, cache=cache, depth=depth,
                   train_pairs=syn + sim + list(also), iters=iters, alpha=alpha)
    coords = np.stack([placed[w] for w in words]).astype(np.float32)
    say(f"core placed: close edges syn={len(syn)} sim={len(sim)} also_see={len(also)}")

    # 2) relational store
    ants = antonym_edges_tiered(words)
    genus_edges = sorted(set(signal_genus.extras(vocab, g)["close_extra"]))
    say(f"edges: antonym={len(ants)} "
        f"({sum(1 for *_, t in ants if t != 'satellite_satellite')} in default tiers), "
        f"genus={len(genus_edges)}")

    # 3) the full sense layer, grounded per-gloss
    from ..wordnet import all_senses
    axis_index = {a: j for j, a in enumerate(axes.names)}
    sense_ids: List[str] = []
    sense_lemmas: List[List[str]] = []
    indptr = [0]
    idxs: List[int] = []
    vals: List[float] = []
    for k, (sid, gloss, lex, lemmas) in enumerate(all_senses()):
        if max_senses is not None and k >= max_senses:
            break
        sense_ids.append(sid)
        sense_lemmas.append(lemmas)
        row: Dict[int, float] = {}
        for prime, wgt in sense_prime_weights(gloss, depth=sense_depth).items():
            j = axis_index.get(prime)
            if j is not None:
                row[j] = wgt
        j = axis_index.get(f"lex:{lex}")
        if j is not None:
            row[j] = 1.0
        for j in sorted(row):
            idxs.append(j)
            vals.append(row[j])
        indptr.append(len(idxs))
        if log and (k + 1) % 20_000 == 0:
            say(f"senses grounded: {k + 1}")
    say(f"senses: {len(sense_ids)} (nnz={len(idxs)})")

    meta = {
        "schema": SCHEMA_VERSION,
        "n_core": n_core,
        "signal_set": ["synonym", "similar", "also_see", "domain-features",
                       "satellite-antonyms(tiered)", "genus(directed)"],
        "counts": {"core_words": len(words), "axes": axes.dim,
                   "senses": len(sense_ids), "antonym_edges": len(ants),
                   "genus_edges": len(genus_edges)},
        "provenance": "M17-M28.1; see RESEARCH_NOTES M29",
    }
    fp = hashlib.sha256(json.dumps(
        {"axes": list(axes.names), "meta": meta}, sort_keys=True).encode()).hexdigest()[:16]
    return USVS(version=SCHEMA_VERSION, fingerprint=fp, axes=list(axes.names),
                core_words=words, core_coords=coords, sense_ids=sense_ids,
                sense_lemmas=sense_lemmas,
                sense_indptr=np.asarray(indptr, dtype=np.int64),
                sense_axis_idx=np.asarray(idxs, dtype=np.int32),
                sense_axis_val=np.asarray(vals, dtype=np.float32),
                antonyms=ants, genus=genus_edges, meta=meta)


# ---------------------------------------------------------------------------
# Persistence (deterministic; three files under a directory)
# ---------------------------------------------------------------------------
def save_usvs(u: USVS, out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "usvs.npz",
        core_words=np.asarray(u.core_words),
        core_coords=u.core_coords,
        sense_ids=np.asarray(u.sense_ids),
        sense_indptr=u.sense_indptr,
        sense_axis_idx=u.sense_axis_idx,
        sense_axis_val=u.sense_axis_val,
    )
    with gzip.open(out / "usvs_senses.json.gz", "wt", encoding="utf-8") as f:
        json.dump(u.sense_lemmas, f, separators=(",", ":"))
    (out / "usvs_meta.json").write_text(json.dumps({
        "version": u.version, "fingerprint": u.fingerprint, "axes": u.axes,
        "antonyms": u.antonyms, "genus": u.genus, "meta": u.meta,
    }, indent=1, sort_keys=True))
    return out


def load_usvs(in_dir) -> USVS:
    p = Path(in_dir)
    z = np.load(p / "usvs.npz", allow_pickle=False)
    m = json.loads((p / "usvs_meta.json").read_text())
    with gzip.open(p / "usvs_senses.json.gz", "rt", encoding="utf-8") as f:
        sense_lemmas = json.load(f)
    return USVS(version=m["version"], fingerprint=m["fingerprint"], axes=m["axes"],
                core_words=[str(w) for w in z["core_words"]],
                core_coords=z["core_coords"],
                sense_ids=[str(s) for s in z["sense_ids"]],
                sense_lemmas=sense_lemmas,
                sense_indptr=z["sense_indptr"], sense_axis_idx=z["sense_axis_idx"],
                sense_axis_val=z["sense_axis_val"],
                antonyms=[tuple(e) for e in m["antonyms"]],
                genus=[tuple(e) for e in m["genus"]], meta=m["meta"])


def export_dictionary(u: USVS, path) -> int:
    """The human-browsable English sense->coordinate dictionary (JSONL.gz)."""
    from ..wordnet import all_senses
    glosses = {sid: gl for sid, gl, _, _ in all_senses()}
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for sid, lemmas in zip(u.sense_ids, u.sense_lemmas):
            f.write(json.dumps({
                "sense": sid, "lemmas": lemmas, "gloss": glosses.get(sid, ""),
                "axes": u.sense_signature(sid),
            }, separators=(",", ":")) + "\n")
            n += 1
    return n
