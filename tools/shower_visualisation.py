#!/usr/bin/env python3
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# ── PARAMÈTRES ───────────────────────────
files_info = {
    "Kaon":   "/gridgroup/ilc/midir/Timing/files/root/100k_kaon_E1to100_digitized.root",    
    "Pion":   "/gridgroup/ilc/midir/Timing/files/root/100k_pi_E1to100_digitized.root",
    "Proton": "/gridgroup/ilc/midir/Timing/files/root/100k_proton_E1to100_digitized.root"
}
treename = "tree"

evt_idx = random.randrange(100000)
# ── PRÉPARATION FIGURE MATPLOTLIB ────────
fig = plt.figure(figsize=(18, 6), constrained_layout=True)

# ── BOUCLE SUR CHAQUE TYPE DE PARTICULE ──
for idx, (label, filename) in enumerate(files_info.items(), 1):
    f = ROOT.TFile.Open(filename)
    tree = f.Get(treename)
    n_tot = tree.GetEntries()

    # Choix d’un événement aléatoire
    tree.GetEntry(evt_idx)

    # Récupération des hits et des données
    xs  = np.array([tree.x[i]   for i in range(tree.x.size())])
    ys  = np.array([tree.y[i]   for i in range(tree.y.size())])
    zs  = np.array([tree.z[i]   for i in range(tree.z.size())])
    ths = np.array([int(tree.thr[i]) for i in range(tree.thr.size())])
    event_number = tree.eventNumber  

    # Ajout d’un subplot 3D
    ax = fig.add_subplot(1, 3, idx, projection='3d')
    colors = ["gray" if t == 1 else "yellow" if t == 2 else "red" for t in ths]

    ax.scatter(xs, ys, zs, c=colors, s=10, depthshade=True)
    ax.set_title(f"{label} – eventNumber {event_number}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")


plt.suptitle("Hadronic showers for three types of particles with the same energy", fontsize=16)
plt.savefig(f'showers_plot/Hadronic_showers_sample_{evt_idx}.png')
plt.show()
