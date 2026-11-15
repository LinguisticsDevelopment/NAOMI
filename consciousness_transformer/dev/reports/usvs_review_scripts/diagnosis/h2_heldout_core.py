"""Re-run the harness placement at 10k winners-set, train_frac 0.5, and SAVE held-out placed coords + test pair splits
(scratch only) so the core-level profile can be computed on genuinely held-out pairs."""
import sys, time, pickle
sys.path.insert(0, "/home/user/NAOMI/consciousness_transformer/src")
import numpy as np
from nsm_ct.ground.ablation import pair_sets, with_features, _in_vocab
from nsm_ct.ground.closeness import split_pairs
from nsm_ct.ground.axes import MeaningAxes
from nsm_ct.ground.cache import DecompCache
from nsm_ct.ground.corpus import gloss_vocabulary
from nsm_ct.ground.placement import place
from nsm_ct.ground.relations import RelationGraph
from nsm_ct.ground import signal_also_see, signal_domain
S = "/tmp/claude-0/-home-user-NAOMI/664d9faf-b57c-55b6-97de-b37fd7e015b5/scratchpad/usvs_diag"
vocab = list(gloss_vocabulary(10000)); g = RelationGraph.build(vocab)
g = with_features(g, signal_domain.extras(vocab, g)["feature_extra"])
close = signal_also_see.extras(vocab, g)["close_extra"]
ax = MeaningAxes.assemble(g, min_attribute_freq=2)
cache = DecompCache(depth=3).warm(vocab)
words = list(vocab); wset = set(words)
ps = pair_sets(words, g)
train_syn, test_syn = split_pairs(ps["syn"], 0.5); train_sim, test_sim = split_pairs(ps["sim"], 0.5)
_, test_ant = split_pairs(ps["ant"], 0.5); train_isa, test_isa = split_pairs(ps["isa"], 0.5)
scored = set(test_syn) | set(test_sim) | set(test_ant) | {tuple(sorted(p)) for p in test_isa}
extra = [p for p in _in_vocab(close, wset) if p not in scored]
placed = place(words, g, ax, cache=cache, depth=3, train_pairs=train_syn + train_sim + extra, iters=20, alpha=0.7)
C = np.stack([placed[w] for w in words]).astype(np.float32)
pickle.dump(dict(words=words, C=C, axes=list(ax.names), test_syn=test_syn, test_sim=test_sim, test_ant=test_ant,
                 test_isa=test_isa, all_syn=ps["syn"], all_ant=ps["ant"], all_isa=ps["isa"]), open(S + "/heldout_core.pkl", "wb"))
print("saved", C.shape, len(test_syn), len(test_ant), len(test_isa))
