import sys, json, pickle, random, os
import numpy as np
sys.path.insert(0, "/home/user/NAOMI/consciousness_transformer/src")
S = "/tmp/claude-0/-home-user-NAOMI/664d9faf-b57c-55b6-97de-b37fd7e015b5/scratchpad/usvs_diag"
DATA = "/home/user/NAOMI/consciousness_transformer/data/usvs"
def load():
    from nsm_ct.ground.usvs import load_usvs
    return load_usvs(DATA, extended=False)
def dense_senses(u):
    n = len(u.sense_ids); d = len(u.axes)
    M = np.zeros((n, d), dtype=np.float32)
    rows = np.repeat(np.arange(n), np.diff(u.sense_indptr))
    M[rows, u.sense_axis_idx] = u.sense_axis_val
    return M
def idf_from(M):
    n = M.shape[0]; df = (M != 0).sum(0)
    return np.log((n + 1.0) / (df + 1.0))
def kinds(axes):
    return np.array(["attr" if a.startswith("attr:") else "lex" if a.startswith("lex:") else "prime" for a in axes])
def wcos(A, B, w=None, mask=None):
    """row-wise cosine of A[i],B[i] with axis weights w (dot = sum w a b) restricted to mask"""
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    ww = np.ones(A.shape[1]) if w is None else np.asarray(w, dtype=np.float64).copy()
    if mask is not None: ww = ww * mask
    num = (A * B * ww).sum(1); na = (A * A * ww).sum(1); nb = (B * B * ww).sum(1)
    den = np.sqrt(na * nb)
    return np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie); rank-based"""
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    r = _rank(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def _rank(x):
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); avg = cs - (cnt - 1) / 2.0
    return avg[inv]
