from evalib import *
sids=B["sids"]
a=E["ant"]; sat=np.array([sids[i].split(".")[-2]=="s" for i in range(N)])
anysat=sat[a[:,0]]|sat[a[:,1]]
print("ant pairs",len(a),"with satellite endpoint",anysat.sum(), "both sat",(sat[a[:,0]]&sat[a[:,1]]).sum())
te=split["ant"]
print("test pairs with satellite endpoint",(anysat&te).sum(),"of",te.sum())
# test antonym pairs: how many senses of each pair have another antonym pair in TRAIN (same synset participates)
tr=a[~te]; trs=set(tr.ravel().tolist())
print("test pairs where either endpoint synset appears in a train antonym pair:",sum((x in trs) or (y in trs) for x,y in a[te]))
print("test pairs where both endpoints appear in train pairs:",sum((x in trs) and (y in trs) for x,y in a[te]))
