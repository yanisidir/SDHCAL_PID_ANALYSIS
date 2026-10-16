# -*- coding: utf-8 -*-
import uproot
import awkward as ak

# Charger le fichier ROOT
file = uproot.open("data/10k_Params.root")
tree = file["paramsTree"]  # Remplace par le vrai nom du TTree

# Lire toutes les données
branches = tree.arrays()
n = len(branches)

# Taille d’un bloc (~10%)
chunk_size = n // 10

# Découpage en 10 fichiers
for i in range(10):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < 9 else n  

    part = branches[start:end]

    output_name = f"part{i+1}.root"
    with uproot.recreate(f'data/{output_name}') as f:
        f["tree"] = part

    print(f"{output_name} sauvegardé ({end - start} événements)")
