#!/usr/bin/env python3
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # pour la projection 3D

# ── PARAMÈTRES ───────────────────────────
filename = "data/100k_pi_merged_digitized.root"  # adapte le chemin si besoin
treename = "tree"
n_events = 500                 # nombre d'événements à moyenniser

# ── OUVERTURE DU FICHIER ROOT ────────────
f    = ROOT.TFile.Open(filename)
tree = f.Get(treename)
n_tot = tree.GetEntries()
n    = min(n_events, n_tot)

# ── DICTIONNAIRES DE COMPTAGE PAR SEUIL ───
# pos_counts_thr[thr] est un dict {(x,y,z): count}
pos_counts_thr = {1: {}, 2: {}, 3: {}}

# ── CALCUL DES COMPTES PAR POSITION ──────
pos_counts = {}
for i in range(n):
    tree.GetEntry(i)
    # on suppose que x,y,z sont std::vector<float> branchés 1D
    xs = tree.x
    ys = tree.y
    zs = tree.z

    # xs = tree.I
    # ys = tree.J
    # zs = tree.K

    ths = tree.thr

    # for j in range(xs.size()):
    #     xi = xs[j]
    #     yi = ys[j]
    #     zi = zs[j]
    #     # arrondir pour regrouper les voisins
    #     key = (round(xi,2), round(yi,2), round(zi,2))
    #     pos_counts[key] = pos_counts.get(key, 0) + 1

    for j in range(xs.size()):
        thr_j = int(ths[j])
        key = (round(xs[j],2), round(ys[j],2), round(zs[j],2))
        d = pos_counts_thr[thr_j]
        d[key] = d.get(key, 0) + 1    

# # ── CONVERSION EN TABLEAUX NUMPY ─────────
# coords = np.array(list(pos_counts.keys()))    # (N_positions, 3)
# counts = np.array(list(pos_counts.values()))  # (N_positions,)
# occup  = counts / n                           # occupation moyenne

# # ── AFFICHAGE 3D ───────────────────────────
# fig = plt.figure(figsize=(8,6))
# ax  = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(
#     coords[:,0], coords[:,1], coords[:,2],
#     c=occup,
#     s=occup * 50,       # échelle arbitraire pour la taille
#     cmap="viridis",
#     depthshade=True
# )
# ax.set_xlabel("x (mm)")
# ax.set_ylabel("y (mm)")
# ax.set_zlabel("z (mm)")
# plt.title(f"Occupation moyenne sur {n} événements")
# plt.colorbar(sc, label="Occupation moyenne")
# plt.tight_layout()
# plt.show()

# ── PRÉPARATION DU PLOT ────────────────────
fig = plt.figure(figsize=(18,6))

for idx, thr_val in enumerate([1,2,3], start=1):
    # Extraction numpy
    coords = np.array(list(pos_counts_thr[thr_val].keys()))
    counts = np.array(list(pos_counts_thr[thr_val].values()))
    occup  = counts / n
    
    ax = fig.add_subplot(1, 3, idx, projection='3d')
    sc = ax.scatter(
        coords[:,0], coords[:,1], coords[:,2],
        c=occup,
        s=occup * 50,
        cmap="viridis",
        depthshade=True
    )
    ax.set_title(f"Seuil thr = {thr_val}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    plt.colorbar(sc, ax=ax, shrink=0.6, label="Occ. moyenne")

plt.suptitle(f"Occupation moyenne par seuil sur {n} événements", y=0.92)
plt.tight_layout()
plt.show()
