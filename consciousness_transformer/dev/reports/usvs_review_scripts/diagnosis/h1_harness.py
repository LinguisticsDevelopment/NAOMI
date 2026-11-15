"""Original held-out harness (ground/ablation.py metric_table), read-only.
Usage: python h1_harness.py N   (N=3000 baseline, 10000 = production-scale winners set)"""
import sys, time
sys.path.insert(0, "/home/user/NAOMI/consciousness_transformer/src")
from nsm_ct.ground.ablation import metric_table, baseline_table, with_features
from nsm_ct.ground.axes import MeaningAxes
from nsm_ct.ground.cache import DecompCache
from nsm_ct.ground.corpus import gloss_vocabulary
from nsm_ct.ground.relations import RelationGraph
from nsm_ct.ground import signal_also_see, signal_domain
N = int(sys.argv[1]); winners = len(sys.argv) > 2 and sys.argv[2] == "winners"
t0 = time.time()
vocab = list(gloss_vocabulary(N)); g = RelationGraph.build(vocab)
close = ()
if winners:
    g = with_features(g, signal_domain.extras(vocab, g)["feature_extra"])
    close = signal_also_see.extras(vocab, g)["close_extra"]
ax = MeaningAxes.assemble(g, min_attribute_freq=2)
cache = DecompCache(depth=3).warm(vocab)
print(f"N={N} winners={winners} vocab={len(vocab)} axes={len(ax.names)} build {time.time()-t0:.0f}s", flush=True)
for j in (0.45, 0.5, 0.55):
    t = metric_table(vocab, g, ax, cache=cache, train_frac=j, close_extra=close)
    print(j, {k: (round(v,3) if isinstance(v,float) else v) for k,v in t.items()}, flush=True)
print(f"total {time.time()-t0:.0f}s")
