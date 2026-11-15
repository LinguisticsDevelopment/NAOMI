from glib import *
for hypw in (0.3, 0.0, 0.05):
    I, J, Wt = graph_edges(hypw)
    lab, g, deg = giant(I, J)
    ok = pos != "?"; ing = lab == g
    print("hypw", hypw, "edges", len(I), "neg", int((Wt < 0).sum()), "giant", int((ing & ok).sum()), "outside", int((~ing & ok).sum()),
          "deg0", int(((deg == 0) & ok).sum()), "ncomp", len(np.unique(lab[ok])), {p: round(float((ing & (pos == p)).sum() / (pos == p).sum()), 3) for p in "nvar"})
