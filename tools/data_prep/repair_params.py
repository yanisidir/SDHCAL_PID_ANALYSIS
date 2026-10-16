import uproot

# Charger le fichier (malgré les erreurs ROOT possibles)
try:
    with uproot.open("130k_proton_E1to130_params.root") as f_in:
        tree = f_in["paramsTree"]
        arrays = tree.arrays()
except Exception as e:
    print("Erreur lors de la lecture du fichier :", e)
    exit(1)

# Sauvegarde dans un fichier propre
with uproot.recreate("130k_proton_E1to130_params_clean.root") as f_out:
    f_out["paramsTree"] = arrays

print("✅ Fichier ROOT propre sauvegardé : 130k_proton_E1to130_params_clean.root")
