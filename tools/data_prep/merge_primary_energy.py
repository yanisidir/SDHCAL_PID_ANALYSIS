import uproot
import awkward as ak
import numpy as np
import os
from collections import defaultdict

def merge_by_key(params_file_path, energy_file_path=None, output_file_path=None,
                 branches_to_merge=("primaryEnergy", "primaryID"),
                 key_branch="eventNumber"):
    """
    Fusionne des branches du RAW vers le PARAMS en alignant sur `key_branch`.
    GÈRE LES DOUBLONS: apparie par ordre d'apparition (i-ème avec i-ème).
    Ne garde QUE les événements présents dans PARAMS.
    """

    # Déduire les chemins par défaut si non fournis
    if energy_file_path is None:
        base = os.path.basename(params_file_path)
        base_no_params = base.replace("_params", "")
        energy_file_path = params_file_path.replace("/params/", "/raw/").replace(base, base_no_params)

    if output_file_path is None:
        root, ext = os.path.splitext(params_file_path)
        output_file_path = f"{root}_merged{ext}"

    with uproot.open(params_file_path) as f_params, uproot.open(energy_file_path) as f_energy:
        if "tree" not in f_params or "tree" not in f_energy:
            raise KeyError("TTree 'tree' manquant dans un des fichiers")
        t_params = f_params["tree"]
        t_energy = f_energy["tree"]

        # Lire tout PARAMS
        data_params = t_params.arrays(library="ak")
        if key_branch not in data_params.fields:
            raise KeyError(f"Branche clé '{key_branch}' absente dans {params_file_path}::tree")

        # Branches RAW disponibles ?
        energy_keys = set(t_energy.keys())
        branches_present = [b for b in branches_to_merge if b in energy_keys]
        branches_missing = [b for b in branches_to_merge if b not in energy_keys]
        if branches_missing:
            print(f"⚠️  Branches absentes dans RAW (ignorées): {branches_missing}")
        if not branches_present:
            raise KeyError(f"Aucune des branches {branches_to_merge} n'existe dans {energy_file_path}::tree")

        # Lire clé + branches RAW
        raw_needed = [key_branch] + branches_present
        data_energy = t_energy.arrays(raw_needed, library="ak")

        # Cast en types stables (évite int32 vs int64)
        params_keys = ak.to_numpy(ak.values_astype(data_params[key_branch], np.int64))
        raw_keys    = ak.to_numpy(ak.values_astype(data_energy[key_branch], np.int64))

        # Index des occurrences: clé -> [positions...]
        pos_params = defaultdict(list)
        for i, k in enumerate(params_keys):
            pos_params[k].append(i)

        pos_energy = defaultdict(list)
        for j, k in enumerate(raw_keys):
            pos_energy[k].append(j)

        # Préparer buffers de sortie pour chaque branche à merger
        n_params = len(params_keys)
        out = {b: [None]*n_params for b in branches_present}

        # Stats
        n_keys = 0
        n_matched = 0
        n_extra_params = 0
        n_extra_energy = 0

        for k, idxP in pos_params.items():
            idxE = pos_energy.get(k, [])
            n_keys += 1
            m = min(len(idxP), len(idxE))
            # apparier i-ème avec i-ème
            for r in range(m):
                ip = idxP[r]
                ie = idxE[r]
                for b in branches_present:
                    out[b][ip] = data_energy[b][ie]
            n_matched += m
            if len(idxP) > len(idxE):
                n_extra_params += (len(idxP) - len(idxE))  # manques côté RAW
            elif len(idxE) > len(idxP):
                n_extra_energy += (len(idxE) - len(idxP))  # surplus côté RAW (OK mais non utilisé)

        print(f"✅ Fusion par '{key_branch}' terminée.")
        print(f"   Clés distinctes rencontrées (PARAMS): {len(pos_params)}")
        print(f"   Appariments réalisés (événements):   {n_matched}")
        print(f"   Manques côté RAW pour PARAMS:        {n_extra_params}")
        print(f"   Surplus côté RAW (non utilisés):     {n_extra_energy}")

        # Injecter dans data_params
        for b in branches_present:
            data_params[b] = ak.Array(out[b])

        # Écriture
        with uproot.recreate(output_file_path) as f_out:
            f_out["tree"] = {name: data_params[name] for name in data_params.fields}

        print(f"➡️  Écrit: {output_file_path}")



def main():
    params_paths = [ 
        "/gridgroup/ilc/midir/analyse/data/params/200k_pi-_1-130_params.root",
        "/gridgroup/ilc/midir/analyse/data/params/200k_proton_1-130_params.root",
    ]

    # params_paths = [ 
    #     "/gridgroup/ilc/midir/analyse/data/params/100k_pi-_5to100_discret_params.root",
    #     "/gridgroup/ilc/midir/analyse/data/params/100k_proton_5to100_discret_params.root",
    # ]

    for params_path in params_paths:
        print(f"\n--- Traitement de {params_path} ---")
        merge_by_key(
            params_path,
            branches_to_merge=("primaryEnergy", "primaryID"),
            key_branch="eventNumber",     # <- adapte si ta clé s'appelle autrement
        )

if __name__ == "__main__":
    main()
