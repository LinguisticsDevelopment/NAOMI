from common import *
from nltk.corpus import wordnet as wn
import collections
heads=[s for s in wn.all_synsets('a')]
hasant={h.name() for h in heads if any(l.antonyms() for l in h.lemmas())}
sat=collections.defaultdict(list)
for s in wn.all_synsets('s'):
    for h in s.similar_tos(): sat[h.name()].append(s.name())
print("heads with sats", len(sat), "of which with direct antonym", sum(1 for h in sat if h in hasant), "without", sum(1 for h in sat if h not in hasant))
print("satellites whose head has direct antonym", sum(len(v) for h,v in sat.items() if h in hasant), "of", sum(len(v) for v in sat.values()))
print("heads without antonym and without satellites", sum(1 for h in heads if h.name() not in hasant and h.name() not in sat))
# heads w/o antonym that are linked to antonym-bearing heads via similar_to? (dumbbell: satellites belong to head with antonym)
print("heads with antonym AND no satellites", sum(1 for h in hasant if h not in sat))
# idf check on axes with base df>0 and unaffected by fn senses
from nsm_ct.ground.usvs import load_usvs
from naomi.score import axis_weights
u=load(); M=dense_senses(u); w=idf_from(M)
ue=load_usvs(DATA, extended=True); wd=axis_weights(ue)[:len(u.axes)]
d=np.abs(wd-w); df=(M!=0).sum(0)
print("max diff over axes with base df>0:", d[df>0].max(), "argmax", u.axes[int(np.argmax(np.where(df>0,d,0)))], "max diff axes df==0", d[df==0].max(), "n axes with diff>0.01:", int((d>0.01).sum()), [u.axes[i] for i in np.where(d>0.01)[0]])
# core words with few axes
C=u.core_coords; nc=(np.abs(C)>1e-6).sum(1)
print("core words with <=10 nonzero axes:", float((nc<=10).mean()), int((nc<=10).sum()), " >=400:", float((nc>=400).mean()))
print("core sparse words examples", [u.core_words[i] for i in np.where(nc<=10)[0][:15]])
print("core min/max rowwise abs max", float(np.abs(C).max(1).min()), float(np.abs(C).max(1).max()))
