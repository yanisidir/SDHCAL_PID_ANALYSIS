#!/usr/bin/env python3
# NN_LGBM_PID_binary_pi_p.py — Inference with LightGBM (BDT) for PID π vs p (sans kaons)

from __future__ import annotations
import logging
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import uproot
import lightgbm as lgb  # LightGBM for classification
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Sequence

# ------------------------------ Logging --------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------ Config ---------------------------------------
class InferenceCfg:
    tree_name = "tree"

    # Features utilisées pour le classif (doivent matcher l'entraînement)
    clf_features = [
        "Thr1", "Thr2", "Thr3", 
        "Begin", "meanRadius", "nMipCluster", "first5LayersRMS",
        "Density", "NClusters", "ratioThr23", "Zbary", "Zrms",
        "PctHitsFirst10", "AvgClustSize", "MaxClustSize", "lambda1", "lambda2",
        "tMin", "tMax", "tMean", "tSpread", "Nmax", "Xmax",
        "eccentricity3D", "transverseRatio", "nTrack"
    ]
    # Features utilisées pour la régression (doivent matcher l'entraînement)
    reg_features = [
        "nHough",
        "nCluster",
        "nMipCluster",
        "nTrack",
        "end",
        "density",
        "transverseRatio",
        "reconstructedCosTheta",
        "meanRadius",
        "first5LayersRMS",
        "propLastLayers",
        "nhitDensity",
        "nHitCustom",
        "nHit2Custom",
        "nHit3Custom",
        "Begin",
        "Radius",
        "Density",
        "ratioThr23",
        "Zbary",
        "Zrms",
        "PctHitsFirst10",
        "PlanesWithClusmore2",
        "MaxClustSize",
        "lambda1",
        "lambda2",
        "N3",
        "tMax",
        "tMean",
        "Nmax",
        "Xmax",
        "sumThrTotal",
        "eccentricity3D",
    ]

    # Colonnes vérité (présentes dans validation_set_pion_proton.root)
    energy_col = "primaryEnergy"
    pdg_col    = "primaryID"

    # ### CHANGEMENT: on ne garde que π et p
    label_to_tag  = {0: "pi", 1: "proton"}          # plus de kaon
    label_to_name = {0: "π-", 1: "p"}               # plus de K0
    pdg_to_label  = {-211: 0, 2212: 1}              # plus de 311

    # --- Smearing temporel optionnel (pour reproduire la résolution) ---
    time_smear_sigma_clf: float | None = 0.1
    time_smear_sigma_reg: float | None = 0.1
    time_cols_clf = ["tMin", "tMax", "tSpread"]
    time_cols_reg = ["tMin"]

    # --- Clip de sécurité en log-espace avant exponentiation ---
    logE_min = np.log(1e-6)
    logE_max = np.log(1e6)


# ------------------------------ Helpers --------------------------------------
def _require_exists(p: Path, what: str):
    # Vérifie si le chemin 'p' existe.
    # Si ce n’est pas le cas, on logge une erreur et on arrête le programme.
    if not p.exists():
        logger.error("%s introuvable: %s", what, p)
        sys.exit(1)

def _require_columns(df: pd.DataFrame, cols: list[str], where: str):
    # Vérifie que le DataFrame 'df' contient bien toutes les colonnes listées dans 'cols'.
    # Si des colonnes manquent, on logge une erreur avec l’endroit concerné ('where') et on arrête le programme.
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.error("Colonnes manquantes (%s): %s", where, ", ".join(missing))
        sys.exit(1)

def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 2) -> np.ndarray:
    # Construit une matrice de confusion pour évaluer les performances d’un classifieur.
    # y_true : étiquettes réelles
    # y_pred : étiquettes prédites
    # n_classes : nombre de classes possibles
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        # On incrémente la case correspondant à (classe réelle, classe prédite)
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm

def _sigma_over_E(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Calcule la résolution en énergie définie par sigma/E.
    # r = (E_reconstruit - E_vrai) / E_vrai
    # La fonction retourne la racine carrée de la moyenne des r^2 (RMS relatif).
    r = (y_pred - y_true) / y_true
    return float(np.sqrt(np.mean(r**2)))


# ------------------------------- I/O -----------------------------------------
# ------------------------------- I/O -----------------------------------------
def smear_columns_gaussian(df: pd.DataFrame, cols: list[str], sigma: float | None, seed: int = 42) -> pd.DataFrame:
    # Applique un "smearing" gaussien (bruit aléatoire gaussien) aux colonnes sélectionnées d’un DataFrame.
    # df     : DataFrame d’entrée
    # cols   : colonnes sur lesquelles appliquer le bruit
    # sigma  : écart-type du bruit gaussien (None => pas de bruit appliqué)
    # seed   : graine pour le générateur de nombres aléatoires (pour reproductibilité)

    if sigma is None:
        # Si aucun sigma n’est fourni, on ne modifie pas le DataFrame
        return df

    rng = np.random.default_rng(seed)  # Générateur aléatoire avec graine fixée
    n = len(df)  # Nombre de lignes (événements)

    for c in cols:
        if c in df.columns:
            # On ajoute un bruit gaussien centré en 0, de variance sigma^2, à la colonne
            df[c] = df[c].to_numpy() + rng.normal(0.0, sigma, size=n)
        else:
            # Si la colonne n’existe pas, on prévient via un warning
            logger.warning("Colonne %s absente pour smearing (ignorée)", c)

    return df

def load_root_files(paths, tree, columns):
    # Charge plusieurs fichiers ROOT et retourne leur contenu sous forme d’un DataFrame pandas.
    # paths   : liste des chemins des fichiers ROOT
    # tree    : nom de l’arbre à ouvrir dans chaque fichier
    # columns : colonnes à extraire depuis l’arbre

    frames = []  # Liste des DataFrames temporaires

    for path in paths:
        p = Path(path)
        if not p.exists():
            # Si le fichier n’existe pas, on logge une erreur et on passe au suivant
            logger.error("Fichier introuvable: %s", path)
            continue

        with uproot.open(p) as f:
            if tree not in f:
                # Si l’arbre demandé n’existe pas, on logge et on stoppe le programme
                logger.error("Tree '%s' absent dans %s", tree, p)
                sys.exit(1)

            # Lecture des colonnes spécifiées et conversion en DataFrame
            frames.append(f[tree].arrays(columns, library="pd"))

    if not frames:
        # Si aucun fichier valide n’a pu être chargé, on arrête le programme
        logger.error("Aucun fichier ROOT valide n'a été chargé.")
        sys.exit(1)

    # Fusion des DataFrames (concaténation ligne par ligne)
    df = pd.concat(frames, ignore_index=True)

    # Nettoyage des valeurs infinies et manquantes
    before = len(df)
    df = df.replace([np.inf, -np.inf], pd.NA).dropna().reset_index(drop=True)

    # Log du nombre d’événements chargés et supprimés
    logger.info("Chargé %d événements (après nettoyage, -%d)", len(df), before - len(df))

    return df

# ------------------------------ Confusion matrix -----------------------------
def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], out: Path):
    # Affiche et sauvegarde une matrice de confusion normalisée en pourcentages.
    # cm     : matrice de confusion brute (entiers)
    # labels : noms des classes (ex. ["π", "p"])
    # out    : chemin du fichier image de sortie (PNG, PDF, etc.)

    # Conversion en pourcentages par rapport au total de chaque ligne (classe réelle)
    pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0

    # Création des annotations (chaque case contient la valeur en % avec 1 décimale)
    annot = np.array([[f"{v:.1f}%" for v in row] for row in pct])

    # Taille de la figure
    plt.figure(figsize=(6, 5))

    # Heatmap avec seaborn
    sns.heatmap(
        pct,                       # matrice normalisée en %
        annot=annot,               # annotations textuelles
        fmt='',                    # pas de formatage numérique par défaut
        cmap='Blues',              # colormap bleue
        xticklabels=labels,        # étiquettes des colonnes (prédit)
        yticklabels=labels,        # étiquettes des lignes (vrai)
        cbar_kws={'label': 'Pourcentage (%)'}  # barre de couleur avec légende
    )

    # Titres et labels des axes
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.title("Matrice de confusion PID (π vs p)")

    # Mise en page compacte
    plt.tight_layout()

    # Sauvegarde de la figure
    plt.savefig(out, dpi=150)

    # Fermeture de la figure pour libérer la mémoire
    plt.close()



# ------------------------------ Inference ------------------------------------
def run_inference(files, out_csv):
    # Fonction principale d'inférence.
    # Elle effectue :
    #   - la lecture des données ROOT
    #   - la classification binaire (π vs proton) avec un modèle LightGBM
    #   - la régression d'énergie spécifique par particule (π, proton)
    #   - le calcul de métriques de performance
    #   - l’enregistrement des résultats dans un CSV

    cfg = InferenceCfg()  # Configuration des features, colonnes, etc.

    # === chemins ===
    clf_model_path  = Path("/gridgroup/.../lgbm_model_29.joblib")  # modèle classif. LGBM sauvegardé

    # Uniquement les régressseurs π et proton (pas de kaons)
    reg_paths = {
        "pi": (
            Path("/gridgroup/.../lgbm_regressor_pi_32.joblib"),     # modèle régression π
            Path("/gridgroup/.../scaler_pi_32.joblib")              # scaler associé
        ),
        "proton": (
            Path("/gridgroup/.../lgbm_regressor_proton_24.joblib"), # modèle régression proton
            Path("/gridgroup/.../scaler_proton_24.joblib")
        ),
    }

    # Vérification de l'existence des modèles
    _require_exists(clf_model_path,  "Modèle classif LGBM")
    for tag, (mp, sp) in reg_paths.items():
        _require_exists(mp, f"Modèle régression {tag}")
        _require_exists(sp, f"Scaler régression {tag}")

    # Colonnes nécessaires = features (classif + régression) + vérité (E, PDG)
    need_cols = sorted(set(cfg.clf_features).union(cfg.reg_features).union([cfg.energy_col, cfg.pdg_col]))
    df_all = load_root_files(files, cfg.tree_name, need_cols)

    # Vérifications des colonnes essentielles
    _require_columns(df_all, cfg.clf_features, "features classif")
    _require_columns(df_all, cfg.reg_features, "features régression")
    _require_columns(df_all, [cfg.energy_col, cfg.pdg_col], "vérité (E, PDG)")

    # Garder uniquement π et protons (supprimer kaons)
    df_all = df_all[df_all[cfg.pdg_col].isin([-211, 2212])].reset_index(drop=True)
    if len(df_all) == 0:
        logger.error("Après filtrage π/p, aucun événement disponible.")
        sys.exit(1)
    logger.info("Événements conservés (π/p uniquement): %d", len(df_all))

    # --- Smearing temporel optionnel ---
    df_all = smear_columns_gaussian(df_all, cfg.time_cols_clf, cfg.time_smear_sigma_clf, seed=42)
    df_all = smear_columns_gaussian(df_all, cfg.time_cols_reg, cfg.time_smear_sigma_reg, seed=42)

    # --- Classification (LightGBM) ---
    logger.info("Chargement classif LGBM (BDT)")
    clf: lgb.LGBMClassifier = joblib.load(clf_model_path)

    # Préparer la matrice de features
    Xc = df_all[cfg.clf_features].to_numpy(dtype=np.float32, copy=False)
    proba_all = clf.predict_proba(Xc)   # Probabilités pour chaque classe
    classes = np.array(clf.classes_)    # Labels internes du modèle

    # On force une classification binaire: pi=0, proton=1
    wanted = {0, 1}
    have = set(classes.tolist())
    if not wanted.issubset(have):
        logger.error("Le classif ne contient pas les classes nécessaires {0,1}. Présentes: %s", classes)
        sys.exit(1)

    # Indices pour extraire les bonnes colonnes de probas
    idx_pi = int(np.where(classes == 0)[0][0])
    idx_p  = int(np.where(classes == 1)[0][0])

    # Matrice (N,2): [P(pi), P(proton)]
    y_proba_bin = np.stack([proba_all[:, idx_pi], proba_all[:, idx_p]], axis=1)

    # Prédictions (0=π, 1=proton)
    y_pred = np.argmax(y_proba_bin, axis=1)
    class_name = np.vectorize(cfg.label_to_name.get)(y_pred)  # noms associés
    proba_max = y_proba_bin[np.arange(len(y_pred)), y_pred]   # probabilité max retenue

    # --- Charger régressseurs/scalers ---
    regs: dict[str, tuple[object, object]] = {}
    for tag, (mpath, spath) in reg_paths.items():
        regs[tag] = (joblib.load(mpath), joblib.load(spath))

    # --- Régression d’énergie ---
    Xr_full = df_all[cfg.reg_features].to_numpy(dtype=np.float32, copy=False)
    E_pred = np.full(len(df_all), np.nan, dtype=np.float32)

    for label_int, tag in cfg.label_to_tag.items():  # {0:"pi", 1:"proton"}
        mask = (y_pred == label_int)
        n = int(mask.sum())
        if n == 0:
            continue
        reg, sc = regs[tag]               # modèle et scaler associés
        Xr = sc.transform(Xr_full[mask])  # normalisation des features

        # Prédiction de log(E), puis re-projection en énergie
        logE = reg.predict(Xr)
        logE = np.clip(logE, cfg.logE_min, cfg.logE_max)  # bornage
        E_pred[mask] = np.exp(logE).astype(np.float32)

        logger.info("Régression %s: %d événements", tag, n)

    # --- Vérités (labels & énergie) ---
    pdg_true = df_all[cfg.pdg_col].to_numpy()
    E_true   = df_all[cfg.energy_col].to_numpy(dtype=np.float32, copy=False)
    label_true_int = np.vectorize(lambda pdg: cfg.pdg_to_label.get(int(pdg), -1))(pdg_true)
    label_true = np.vectorize(lambda k: cfg.label_to_name.get(int(k), "UNK"))(label_true_int)

    # --- Métriques classification ---
    valid_mask = (label_true_int >= 0)
    if np.any(valid_mask):
        acc = float((y_pred[valid_mask] == label_true_int[valid_mask]).mean())
        cm = _confusion_matrix(label_true_int[valid_mask], y_pred[valid_mask], n_classes=2)
        logger.info("Classification accuracy (π vs p): %.4f", acc)
        logger.info("Confusion matrix:\n%s", cm)

        # Sauvegarde de la matrice de confusion
        class_names = [cfg.label_to_name[i] for i in range(2)]  # ["π-", "p"]
        png_path = "confusion_matrix_pid_LGBM_pi_p.png"
        plot_confusion_matrix(cm, class_names, png_path)
        logger.info("Matrice de confusion sauvegardée: %s", png_path)
    else:
        logger.warning("Aucun label vrai reconnu (PDG inattendus).")

    # --- Métriques régression ---
    if np.all(np.isfinite(E_pred)):
        rmse = mean_squared_error(E_true, E_pred, squared=False)
        mae  = mean_absolute_error(E_true, E_pred)
        r2   = r2_score(E_true, E_pred)
        sigma_rel = _sigma_over_E(E_true, E_pred)
        logger.info("Régression (global): RMSE=%.4f  MAE=%.4f  R²=%.4f  σ(E)/E=%.4f",
                    rmse, mae, r2, sigma_rel)

        # Métriques par particule (π et proton séparés)
        for lbl_int, lbl_name in cfg.label_to_name.items():
            m = (label_true_int == lbl_int)
            if np.any(m):
                rmse_c = mean_squared_error(E_true[m], E_pred[m], squared=False)
                mae_c  = mean_absolute_error(E_true[m], E_pred[m])
                r2_c   = r2_score(E_true[m], E_pred[m])
                s_c    = _sigma_over_E(E_true[m], E_pred[m])
                logger.info("Régression (true=%s): RMSE=%.4f  MAE=%.4f  R²=%.4f  σ(E)/E=%.4f",
                            lbl_name, rmse_c, mae_c, r2_c, s_c)
    else:
        logger.warning("Certaines prédictions d'énergie sont NaN (classes sans routeur régression ?).")

    # --- Résultat final (DataFrame + CSV) ---
    out_df = pd.DataFrame({
        # vérité
        "PDG_true": pdg_true,
        "label_true_int": label_true_int,
        "label_true": label_true,
        "E_true_GeV": E_true,
        # classification
        "label_pred_int": y_pred,
        "label_pred": class_name,
        "proba_pred": proba_max,
        "proba_pi":     y_proba_bin[:, 0],
        "proba_proton": y_proba_bin[:, 1],
        # régression
        "E_pred_GeV": E_pred,
        # debug
        "dE_over_E": (E_pred - E_true) / E_true,
    })

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info("Résultats écrits dans %s (%d lignes)", out_path, len(out_df))



def main():
    files = ["/gridgroup/ilc/midir/analyse/data/validation_set_pion_proton.root"]

    out_csv = "BDT_BDT.csv"

    run_inference(files, out_csv)


if __name__ == "__main__":
    main()
