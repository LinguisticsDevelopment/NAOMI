"""Reproduce the M19.1 dimensionality_spectrum numbers (anchored feature matrix, first 3000 words, 234 axes) and the same at 10k winners axes (607)."""
import sys; sys.path.insert(0, "/home/user/NAOMI/consciousness_transformer/src")
from nsm_ct.ground.axes import MeaningAxes, dimensionality_spectrum
from nsm_ct.ground.cache import DecompCache
from nsm_ct.ground.corpus import gloss_vocabulary
from nsm_ct.ground.relations import RelationGraph
from nsm_ct.ground.ablation import with_features
from nsm_ct.ground import signal_domain
for n, win in ((3000, False), (10000, True)):
    v = list(gloss_vocabulary(n)); g = RelationGraph.build(v)
    if win: g = with_features(g, signal_domain.extras(v, g)["feature_extra"])
    ax = MeaningAxes.assemble(g, min_attribute_freq=2); c = DecompCache(depth=3).warm(v)
    for mw in (3000, 10000):
        if mw > n: continue
        r = dimensionality_spectrum(g, ax, cache=c, max_words=mw)
        print(n, win, "max_words", mw, {k: (round(x,2) if isinstance(x,float) else x) for k, x in r.items() if k != "top_singular_values"}, flush=True)
