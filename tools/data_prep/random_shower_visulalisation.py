#!/usr/bin/env python3
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# ── PARAMÈTRES ───────────────────────────
filename = "data/100k_pi_merged_digitized.root"  # adapte le chemin si besoin
treename = "tree"

# ── OUVERTURE DU FICHIER ROOT ────────────
f    = ROOT.TFile.Open(filename)
tree = f.Get(treename)
n_tot = tree.GetEntries()

# ── CHOIX D’UN ÉVÉNEMENT ALÉATOIRE ───────
evt_idx = random.randrange(n_tot)
tree.GetEntry(evt_idx)

# ── Récupération des hits de l’événement ──
# On suppose que x,y,z sont des std::vector<float>, thr est std::vector<int>
xs  = np.array([tree.x[i]   for i in range(tree.x.size())])
ys  = np.array([tree.y[i]   for i in range(tree.y.size())])
zs  = np.array([tree.z[i]   for i in range(tree.z.size())])
ths = np.array([int(t)      for t in tree.thr])

# ── AFFICHAGE 3D DE L’ÉVÉNEMENT ───────────
fig = plt.figure(figsize=(8,6))
ax  = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    xs, ys, zs,
    c=ths,               # couleur = seuil de déclenchement
    s=10,                # taille fixe ou fonction de thr si tu préfères
    cmap="plasma",
    depthshade=True
)
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_zlabel("z (mm)")
plt.title(f"Gerbe hadronique – événement aléatoire #{evt_idx}")
plt.colorbar(sc, label="Seuil thr")
plt.tight_layout()
plt.show()
