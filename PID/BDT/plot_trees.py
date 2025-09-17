#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Charger un modèle LightGBM entraîné et sauvegarder :

Ce script :
  1) charge un modèle LightGBM (wrapper sklearn) sauvegardé en .joblib,
  2) enregistre l'importance des features (gain + bar),
  3) enregistre des schémas d'arbres en PNG,
  4) calcule et enregistre des graphiques SHAP (summary bar + beeswarm).

    Pour SHAP, tu dois fournir un échantillon X (features seulement, même ordre que l'entraînement).
    Tu peux pointer vers un CSV (X_PATH) : colonnes = noms de features.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt

# SHAP est optionnel ; si absent, on continue sans
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

import pandas as pd
import numpy as np



# ======================
# PARAMÈTRES À MODIFIER
# ======================
MODEL_PATH = "/gridgroup/ilc/midir/analyse/PID/BDT/results_with_time/models/lgbm_model_26.joblib"      # chemin vers ton modèle sauvegardé
OUTPUT_DIR = Path("lgbm_viz")    # dossier de sortie
X_PATH       = "results_with_time/data/X_test_26.csv"  
TREE_INDICES = [0, 1, 2]          # indices des arbres à tracer (ou "all")
TOP_FEATURES = 30                 # top-K features dans l'importance
SHAP_SAMPLE  = 2000               # nb. de lignes max pour SHAP (pour accélérer)
# ======================



def _ensure_feature_order(X_df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Essaie d'aligner l'ordre des colonnes sur celui utilisé à l'entraînement.
    """
    feat_names = getattr(model, "feature_name_", None)
    if feat_names is None:
        # Certains modèles LightGBM côté sklearn stockent les noms dans booster_.feature_name()
        try:
            feat_names = model.booster_.feature_name()
        except Exception:
            feat_names = None

    if feat_names:
        missing = [c for c in feat_names if c not in X_df.columns]
        if missing:
            raise ValueError(
                f"Les features suivantes attendues par le modèle sont absentes du CSV : {missing}"
            )
        X_df = X_df.loc[:, feat_names]
    else:
        # pas d'info — on suppose que l'ordre du CSV correspond à l'entraînement
        pass
    return X_df

def plot_and_save_feature_importance(booster: lgb.Booster, outdir: Path, topk: int):
    # Importance "gain" (barres horizontales LightGBM)
    ax = lgb.plot_importance(
        booster,
        importance_type="gain",
        max_num_features=topk,
        figsize=(10, 8),
    )
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(outdir / "feature_importance_gain.png", dpi=300)
    plt.close(fig)

    # Importance "split" (nombre d'utilisations dans les splits), via SHAP bar si dispo
    try:
        # LightGBM n’a pas de plot direct "split" en bar standard ; restons sur gain pour la clarté.
        pass
    except Exception:
        pass

def plot_and_save_trees(booster: lgb.Booster, outdir: Path, tree_indices):
    if isinstance(tree_indices, str) and tree_indices.lower() == "all":
        tree_indices = list(range(booster.num_trees()))

    for idx in tree_indices:
        ax = lgb.plot_tree(
            booster,
            tree_index=idx,
            figsize=(24, 16),
            show_info=["split_gain", "internal_value", "internal_count", "leaf_count", "leaf_weight", "data_percentage"],
        )
        fig = ax.figure
        fig.tight_layout()
        outpath = outdir / f"tree_{idx:04d}.png"
        fig.savefig(outpath, dpi=300)
        plt.close(fig)

def compute_and_save_shap(model, X: pd.DataFrame, outdir: Path, sample: int):
    """
    Calcule et enregistre :
      - SHAP summary bar (importance moyenne absolue),
      - SHAP summary beeswarm (distribution des effets).
    Gère binaire et multiclasses.
    """
    if not HAS_SHAP:
        print("[!] SHAP non installé : `pip install shap` pour activer ces graphiques.")
        return

    if len(X) > sample:
        X_shap = X.sample(n=sample, random_state=42)
    else:
        X_shap = X.copy()

    # Explainer optimisé pour arbres
    explainer = shap.TreeExplainer(model)
    
    # shap_values peut être :
    # - np.ndarray (binaire/régression), shape (n, n_features)
    # - list de arrays (multiclasses), len = n_classes
    shap_values = explainer.shap_values(X_shap)

    sv_list = [-shap_values, shap_values]     ## 

    # ---- Summary bar ----
    plt.figure()
    shap.summary_plot(sv_list, X_shap, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(outdir / "shap_summary_bar.png", dpi=300)
    plt.close()

    # ---- Beeswarm ----
    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(outdir / "shap_summary_beeswarm.png", dpi=300)
    plt.close()

    print("[OK] SHAP plots sauvegardés (bar + beeswarm).")
    
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Charger le modèle sklearn (.joblib)
    print("[*] Chargement du modèle…")
    model = joblib.load(MODEL_PATH)

    # 2) Récupérer le Booster natif pour les plots LightGBM
    booster = model.booster_

    # 3) Importance des features
    print("[*] Création des graphiques d'importance des features…")
    plot_and_save_feature_importance(booster, OUTPUT_DIR, TOP_FEATURES)
    print(f"    -> {OUTPUT_DIR/'feature_importance_gain.png'}")

    # 4) Visualisation d'arbres
    print("[*] Création des schémas d'arbres…")
    plot_and_save_trees(booster, OUTPUT_DIR, TREE_INDICES)
    print(f"    -> PNG des arbres dans : {OUTPUT_DIR.resolve()}")

    # 5) SHAP (si X disponible)
    if Path(X_PATH).exists():
        print("[*] Chargement des features pour SHAP…")
        X = pd.read_csv(X_PATH)
        X = _ensure_feature_order(X, model)
        print("[*] Calcul et sauvegarde des graphiques SHAP…")
        compute_and_save_shap(model, X, OUTPUT_DIR, SHAP_SAMPLE)
        print(f"    -> {OUTPUT_DIR/'shap_summary_bar.png'}")
        print(f"    -> {OUTPUT_DIR/'shap_summary_beeswarm.png'}")
    else:
        print(f"[!] Fichier features introuvable pour SHAP : {X_PATH}. "
              "Les plots SHAP sont sautés. Fournis un CSV de features si tu veux les générer.")

    print("[OK] Terminé.")


if __name__ == "__main__":
    main()
