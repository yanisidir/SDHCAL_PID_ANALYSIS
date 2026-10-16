#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
root_to_csv_batch.py

Convertit tous les fichiers ROOT contenant un TTree donné en fichiers CSV.
Se limite aux fichiers contenant "digitized" dans le nom.
"""

import uproot
import pandas as pd
import os
from glob import glob


def root_to_csv(root_path: str, tree_name: str, csv_path: str) -> None:
    with uproot.open(root_path) as root_file:
        try:
            tree = root_file[tree_name]
        except KeyError:
            raise KeyError(f"TTree '{tree_name}' non trouvé dans '{root_path}'")

        selected_branches = [
            "eventNumber",
            "particle1PDG",
            "I", "J", "K",
            "x", "y", "z",
            "thr", "time"
        ]

        available = set(tree.keys())
        for b in selected_branches:
            if b not in available:
                raise ValueError(f"Branche manquante : {b} dans {root_path}")

        df = tree.arrays(selected_branches, library="pd")

        df.to_csv(csv_path, index=False)
        print(f"[✓] Exporté : {csv_path}")


if __name__ == "__main__":
    tree_name = "tree"

    # Liste tous les fichiers *.root contenant "digitized"
    root_files = sorted(glob("*digitized.root"))

    if not root_files:
        print("Aucun fichier 'digitized.root' trouvé dans le dossier courant.")
    else:
        for root_file in root_files:
            csv_file = root_file.replace(".root", ".csv")
            try:
                root_to_csv(root_file, tree_name, csv_file)
            except Exception as e:
                print(f"[✗] Échec pour {root_file} : {e}")
