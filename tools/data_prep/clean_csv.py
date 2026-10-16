import pandas as pd
import ast
import os

from sdhcal import paths as sdhcal_paths

def explode_event_csv(input_path, output_path):
    print(f" Lecture du fichier : {input_path}")
    df = pd.read_csv(input_path)

    # Conversion des chaînes en listes Python
    for col in ['x', 'y', 'z', 'thr', 'time']:
        df[col] = df[col].apply(ast.literal_eval)

    # Vérification de cohérence des longueurs
    def is_consistent(row):
        n = len(row['x'])
        return all(len(row[col]) == n for col in ['y', 'z', 'thr', 'time'])

    inconsistent = df[~df.apply(is_consistent, axis=1)]
    if not inconsistent.empty:
        print(f"⚠ {len(inconsistent)} lignes ignorées pour incohérence de tailles.")
        df = df[df.apply(is_consistent, axis=1)]

    # Explosion
    df_exploded = df.explode(['x', 'y', 'z', 'thr', 'time']).copy()

    # Renommer les colonnes pour correspondre au format du GNN
    df_exploded.rename(columns={'x': 'x_coords', 'y': 'y_coords', 'z': 'z_coords'}, inplace=True)

    # Conversion en float
    for col in ['x_coords', 'y_coords', 'z_coords', 'thr', 'time']:
        df_exploded[col] = pd.to_numeric(df_exploded[col])

    # Sauvegarde
    df_exploded.to_csv(output_path, index=False)
    print(f" Fichier plat sauvegardé dans : {output_path}")

if __name__ == "__main__":
    # Rescued from data/data_1k/, where it was living in a git-ignored tree. The
    # internship paths are translated rather than kept: legacy() maps the old
    # machine's /gridgroup prefix onto $SDHCAL_DATA_DIR, so the script runs here.
    LOTS = [
        ("100k_pi_E1to100_digitized.csv", "100k_pi_E1to100_discret_flat.csv"),
        ("100k_proton_E1to100_digitized.csv", "100k_proton_E1to100_flat.csv"),
        ("100k_kaon_E1to100_digitized.csv", "100k_kaon_E1to100_flat.csv"),
        ("130k_pi_E1to130_digitized.csv", "130k_pi_E1to130_discret_flat.csv"),
        ("130k_proton_E1to130_digitized.csv", "130k_proton_E1to130_flat.csv"),
        ("130k_kaon_E1to130_digitized.csv", "130k_kaon_E1to130_flat.csv"),
        ("10k_kaon_E10to100_discret_digitized.csv", "10k_kaon_E10to100_discret_flat.csv"),
    ]

    for source, destination in LOTS:
        explode_event_csv(
            str(sdhcal_paths.legacy("/gridgroup/ilc/midir/Timing/files/csv/" + source)),
            str(sdhcal_paths.legacy("/gridgroup/ilc/midir/Timing/files/csv/" + destination)),
        )
