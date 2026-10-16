import uproot
import awkward as ak
import numpy as np

# === PARAMÈTRES ===
file_paths = [
    "/gridgroup/ilc/midir/analyse/data/merged_primaryEnergy/100k_pi-_5to100_discret_params_merged.root",
    "/gridgroup/ilc/midir/analyse/data/merged_primaryEnergy/100k_proton_5to100_discret_params_merged.root",
]
tree_name = "tree"
output_root_file = "validation_set_pion_proton.root"

# Optionnel : ne lire que certaines branches si tu veux réduire encore la charge
# branches = ["eventNumber", "particle1PDG", "I", "J", "K", "thr", "time"]
branches = None  # None = toutes les branches

# === CHARGEMENT EN AWKWARD (rapide pour jagged) ===
arrays = []
for path in file_paths:
    with uproot.open(path) as f:
        t = f[tree_name]
        arr = t.arrays(branches, library="ak", how="zip")  # enregistre un record {champ1, champ2, ...}
        arrays.append(arr)

# === CONCAT + SHUFFLE ===
arr_all = ak.concatenate(arrays, axis=0)
n = len(arr_all)
perm = np.random.default_rng(42).permutation(n)
arr_all = arr_all[perm]

# === ÉCRITURE ROOT (sans copie inutile) ===
with uproot.recreate(output_root_file) as fout:
    # on "dézippe" le record en dict {branche: données}
    fout[tree_name] = {name: arr_all[name] for name in arr_all.fields}

print(f"Fichier ROOT sauvegardé : {output_root_file} avec {n} entrées mélangées.")
