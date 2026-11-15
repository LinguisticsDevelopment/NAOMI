from evalib import *
import json
sids=B["sids"]; sidx={s:i for i,s in enumerate(sids)}
Mf=np.asarray(M,dtype=np.float64); W=Mf*np.sqrt(IDF)
def vec(i,c,v):  # flattened [w;w ; c; c*v]
    return np.concatenate([W[i],W[i],[c,c*v]])
def cs(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
out={}
for lam in (0.9,3.0):
  for X,O in (("happy.a.01","unhappy.a.01"),("hot.a.01","cold.a.01"),("large.a.01","small.a.01"),("good.a.01","bad.a.01")):
    i,j=sidx[X],sidx[O]; nx=np.linalg.norm(W[i]); no=np.linalg.norm(W[j])
    items=[("X",vec(i,lam*nx,1.0)),("very X",vec(i,lam*nx,1.5)),("slightly X",vec(i,lam*nx,0.3)),("not X",vec(i,lam*nx,-1.0)),("opp(X)",vec(j,lam*no,-1.0))]
    mat=[[round(cs(a,b),3) for _,b in items] for _,a in items]
    out[f"lam{lam}|{X}"]=dict(labels=[n for n,_ in items],mat=mat,share=round(lam**2/(1+lam**2),3))
    print(lam,X,[n for n,_ in items]); [print(r) for r in mat]
json.dump(out,open(P+"p2_degree.json","w"),indent=1)
