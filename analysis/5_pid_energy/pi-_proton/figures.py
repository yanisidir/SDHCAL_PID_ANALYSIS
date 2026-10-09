#!/usr/bin/env python3
# plot_from_csv_ROOT_like_npz.py
# Lit un CSV avec colonnes E_true_GeV, E_pred_GeV (+ label_true_int ou label_true)
# et recrée les 4 figures ROOT suivantes, comme ta macro .npz :
# - plots/Resolution_relative_all_BDT.png
# - plots/Lin_profile_all_BDT.png
# - plots/Relative_deviation_all_BDT.png
# - plots/Lin_n_Dev_all_BDT.png

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Mapping, Optional

import numpy as np
import pandas as pd
import os

import ROOT
from ROOT import (
    TCanvas, TGraphErrors, TLegend, TH1F, TH1D, TLine, TBox, TPad,
    gROOT, gStyle, kRed, kBlue, kGreen
)

# Empêche ROOT d'attacher automatiquement les TH1 à un "directory" global.
# Pratique en Python pour éviter des fuites de mémoire/gestion d'ownership.
ROOT.TH1.AddDirectory(False)

# Mode batch : aucune fenêtre graphique n'est ouverte (utile pour jobs/headless).
gROOT.SetBatch(True)

# Pas de panneau de stats (moyenne, RMS, etc.) sur les histogrammes par défaut.
gStyle.SetOptStat(0)

# --------------------- Style / mapping ---------------------
# Couleurs ROOT par espèce (utilisées pour les TGraphErrors, lignes, etc.)
COL = {"pi": kRed, "proton": kBlue}

# Styles de marqueurs ROOT par espèce (20 = rond plein, 22 = triangle plein).
MRK = {"pi": 20,   "proton": 22}

# Étiquettes Latex ROOT pour la légende (π− et p).
LEG = {"pi": "#pi^{-}", "proton": "p"}

# Mappage de labels entiers → nom canonique (utilisé à l'import CSV).
INT_TO_CANON = {0: "pi", 1: "proton"}

# Normalisation de labels texte → nom canonique.
TEXT_TO_CANON = {
    "pi": "pi", "π-": "pi", "pi-": "pi", "pion": "pi",
    "p": "proton", "proton": "proton",
}

# --------------------- Binning / Fit ---------------------
# Binning fixe en énergie faisceau.
#       fixent E_MIN_FIXED = 5.0 et NBINS_FIXED = w0. À harmoniser si besoin.
E_MIN_FIXED = 5.0
E_MAX_FIXED = 100.0
NBINS_FIXED = 20

# Paramètres de l'histogramme de ΔE = E_reco - E_true pour le fit gaussien.
DELTAE_NBINS = 100
DELTAE_MIN = -20.0
DELTAE_MAX =  20.0

# Seuil minimal d'événements par bin pour tenter un ajustement gaussien.
MIN_EVENTS_FOR_FIT = 10


# -------------------------- Utils --------------------------
def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

# def _bin_edges_fixed(nbins: int = NBINS_FIXED,
#                      emin: float = E_MIN_FIXED,
#                      emax: float = E_MAX_FIXED) -> np.ndarray:
#     return np.linspace(emin, emax, nbins+1)

def _bin_edges_fixed():
    # Bins centrés sur les valeurs générées (5,10,...,100) avec largeur ±2.5
    return np.arange(2.5, 102.5 + 1e-9, 5.0)

def _as_graph(x: np.ndarray, y: np.ndarray,
              ex: Optional[np.ndarray] = None,
              ey: Optional[np.ndarray] = None,
              name: str = "g") -> TGraphErrors:
    """
    Convertit des données (x, y) avec incertitudes optionnelles (ex, ey) 
    en un objet ROOT TGraphErrors.

    Paramètres
    ----------
    x : np.ndarray
        Tableau des coordonnées en x.
    y : np.ndarray
        Tableau des coordonnées en y.
    ex : Optional[np.ndarray], défaut = None
        Erreurs sur x. Si None, un tableau de zéros est utilisé.
    ey : Optional[np.ndarray], défaut = None
        Erreurs sur y. Si None, un tableau de zéros est utilisé.
    name : str, défaut = "g"
        Nom attribué au TGraphErrors.

    Retour
    ------
    TGraphErrors
        Graphique ROOT contenant les points (x, y) avec barres d'erreurs.
    """

    # Nombre de points
    n = len(x)

    # Si aucune erreur en x ou y n'est fournie, on crée des tableaux de zéros
    ex = np.zeros(n) if ex is None else ex
    ey = np.zeros(n) if ey is None else ey

    # Création du graphique ROOT TGraphErrors en convertissant 
    # les données en float64 (format attendu par ROOT)
    g = TGraphErrors(n,
                     x.astype(np.float64), 
                     y.astype(np.float64),
                     ex.astype(np.float64), 
                     ey.astype(np.float64))

    # Attribution d’un nom au graphique
    g.SetName(name)

    # Retourne le graphique construit
    return g

def _style_graph(g: TGraphErrors, color: int, marker: int) -> None:
    """
    Applique un style graphique à un objet ROOT TGraphErrors.

    Paramètres
    ----------
    g : TGraphErrors
        Le graphique ROOT à styliser.
    color : int
        Code couleur ROOT pour la ligne et les marqueurs.
    marker : int
        Style de marqueur ROOT (forme des points).
    """

    # Définit la couleur de la ligne reliant les points
    g.SetLineColor(color)

    # Définit la couleur des marqueurs (points de données)
    g.SetMarkerColor(color)

    # Définit le style du marqueur (ex. rond, carré, triangle)
    g.SetMarkerStyle(marker)
# ----------------------- Chargement CSV ---------------------
def load_csv_grouped(csv_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Lit un fichier CSV et renvoie un dictionnaire des données groupées par particule.
    Le dictionnaire est de la forme :
        { 'pi' | 'kaon' | 'proton' : (E_true, E_pred) }

    Colonnes requises dans le CSV :
        - E_true_GeV : énergie vraie en GeV
        - E_pred_GeV : énergie reconstruite en GeV
    
    La particule est déduite en priorité de la colonne `label_true_int`,
    puis de `label_true` (texte). Si non trouvé → défaut : 'pi'.
    """

    # Vérifie que le fichier existe
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path.resolve()}")

    # Charge le CSV dans un DataFrame pandas
    df = pd.read_csv(csv_path)

    # Vérifie la présence des colonnes obligatoires
    needed = ["E_true_GeV", "E_pred_GeV"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {', '.join(missing)}")

    # Nettoie les valeurs infinies ou NaN, puis crée deux colonnes en float
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    df["E_beam"] = df["E_true_GeV"].astype(float)   # énergie "vraie"
    df["E_reco"] = df["E_pred_GeV"].astype(float)   # énergie reconstruite

    # Détermination du type de particule
    particle = None
    if "label_true_int" in df.columns:
        # Mappe les labels entiers vers un nom canonique (via dictionnaire INT_TO_CANON)
        particle = df["label_true_int"].map(INT_TO_CANON)

    if "label_true" in df.columns:
        if particle is None:
            # Initialise une série vide si label_true_int absent
            particle = pd.Series([np.nan]*len(df), index=df.index)
        # Normalise le texte : minuscule, sans espaces
        lt_norm = df["label_true"].astype(str).str.strip().str.lower()
        # Remplit les valeurs manquantes avec le mapping texte→nom canonique
        particle = particle.fillna(lt_norm.map(TEXT_TO_CANON))

    if particle is None:
        # Ni label_true_int ni label_true → erreur
        raise ValueError("Impossible de déduire la particule (ni label_true_int, ni label_true).")

    # Remplace les NaN restants par 'pi' par défaut
    df["particle"] = particle.fillna("pi")

    # Regroupe les données par particule
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for label in ("pi", "proton"):  # ← ici seuls pi et proton sont considérés
        sub = df[df["particle"] == label]
        if not sub.empty:
            # Stocke les énergies sous forme de tableaux numpy
            out[label] = (
                sub["E_beam"].to_numpy(float),
                sub["E_reco"].to_numpy(float)
            )

    # Si aucune donnée exploitable → erreur
    if not out:
        raise RuntimeError("Aucune ligne utilisable après regroupement par particule.")

    return out

# -------------------- Calculs binned (identiques) --------------------
def _binned_profile(y_true: np.ndarray, y_pred: np.ndarray, bins: np.ndarray):
    """
    Calcule le profil binned (moyenne et erreur-type sur y_pred dans chaque bin de y_true).

    Paramètres
    ----------
    y_true : np.ndarray
        Valeurs vraies (utilisées pour le binning).
    y_pred : np.ndarray
        Valeurs prédites (moyennées par bin de y_true).
    bins : np.ndarray
        Bornes des intervalles de binning (par ex. np.linspace(...)).

    Retour
    ------
    centers : np.ndarray
        Centres des bins (moyenne des bornes inférieure et supérieure).
    mean : np.ndarray
        Moyennes de y_pred dans chaque bin de y_true.
    sem : np.ndarray
        Erreurs standards sur la moyenne (σ / √n) dans chaque bin.
    """

    # Calcule les centres de chaque bin comme la moyenne des bornes
    centers = 0.5 * (bins[1:] + bins[:-1])

    # Initialise des tableaux de même taille que centers
    mean = np.zeros_like(centers)
    sem  = np.zeros_like(centers)

    # Boucle sur les bins définis par les intervalles (lo, hi)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        # Sélectionne les indices des points dont y_true tombe dans [lo, hi)
        m = (y_true >= lo) & (y_true < hi)
        n = int(np.sum(m))  # nombre de points dans le bin

        if n > 0:
            # Sous-échantillon de y_pred correspondant à ce bin
            v = y_pred[m]

            # Moyenne et écart-type (population, ddof=0)
            mu = float(v.mean())
            sd = float(v.std(ddof=0))

            # Stocke la moyenne et l'erreur standard de la moyenne (σ / √n)
            mean[i] = mu
            sem[i]  = sd / np.sqrt(n)
        else:
            # Aucun point dans ce bin → on met 0
            mean[i] = 0.0
            sem[i]  = 0.0

    # Retourne centres, moyennes et erreurs
    return centers, mean, sem

def _binned_deviation_percent(y_true: np.ndarray, y_pred: np.ndarray, bins: np.ndarray):
    """
    Calcule la déviation moyenne en pourcentage entre y_pred et la valeur attendue (centre du bin de y_true),
    ainsi que son incertitude statistique.

    Paramètres
    ----------
    y_true : np.ndarray
        Valeurs vraies utilisées pour le binning.
    y_pred : np.ndarray
        Valeurs prédites, sur lesquelles on calcule la déviation.
    bins : np.ndarray
        Bornes des intervalles de binning (par ex. np.linspace(...)).

    Retour
    ------
    centers : np.ndarray
        Centres des bins (moyenne des bornes inférieure et supérieure).
    dev : np.ndarray
        Déviation moyenne en pourcentage = 100 * (⟨y_pred⟩ - center) / center
    err : np.ndarray
        Erreur standard de la moyenne, exprimée en pourcentage.
    """

    # Calcule les centres des bins
    centers = 0.5 * (bins[1:] + bins[:-1])

    # Initialise les tableaux résultat
    dev = np.zeros_like(centers)
    err = np.zeros_like(centers)

    # Boucle sur chaque bin
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        # Masque des valeurs appartenant à ce bin
        m = (y_true >= lo) & (y_true < hi)
        n = int(np.sum(m))  # nombre de points dans le bin

        if n > 0:
            # Valeurs prédites correspondant à ce bin
            yhat = y_pred[m]

            # Moyenne et écart-type
            mu = float(yhat.mean())
            sd = float(yhat.std(ddof=0))

            # Centre du bin (sécurité : éviter la division par zéro)
            c = float(centers[i]) if centers[i] > 0 else 1.0

            # Déviation moyenne en pourcentage
            dev[i] = 100.0 * (mu - c) / c

            # Erreur standard de la moyenne (σ/√n), exprimée en %
            err[i] = 100.0 * (sd / np.sqrt(n)) / c
        else:
            # Aucun point dans ce bin → valeurs nulles
            dev[i] = 0.0
            err[i] = 0.0

    return centers, dev, err


def _binned_resolution_gaussfit_percent(y_true: np.ndarray, y_pred: np.ndarray, bins: np.ndarray):
    """
    Calcule la résolution énergétique (%) en ajustant un gaussien
    à la distribution de ΔE = (E_reco - E_true) dans chaque bin de E_true.

    Paramètres
    ----------
    y_true : np.ndarray
        Énergies vraies (GeV), utilisées pour le binning.
    y_pred : np.ndarray
        Énergies reconstruites (GeV).
    bins : np.ndarray
        Bornes des bins en énergie vraie.

    Retour
    ------
    centers : np.ndarray
        Centres des bins.
    sig_pct : np.ndarray
        Résolution gaussienne (σ/E) en pourcentage.
    esig_pct : np.ndarray
        Erreur sur la résolution (δσ/E) en pourcentage.
    """

    # Centres des bins
    centers = 0.5 * (bins[1:] + bins[:-1])

    # Tableaux résultats
    sig_pct = np.zeros_like(centers)
    esig_pct = np.zeros_like(centers)

    # Boucle sur chaque bin
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        # Sélection des événements dont E_true tombe dans le bin
        m = (y_true >= lo) & (y_true < hi)

        # Vérifie qu'il y a suffisamment d'événements pour un fit gaussien
        if np.sum(m) < MIN_EVENTS_FOR_FIT:
            sig_pct[i] = 0.0
            esig_pct[i] = 0.0
            continue

        # ΔE = E_reco - E_true
        dE = (y_pred[m] - y_true[m]).astype(np.float64)

        # Histogramme ROOT de ΔE (par ex. dans [-20,20] GeV)
        hname = f"hde_bin{i}"
        h = TH1D(
            hname,
            "DE;E_{reco}-E_{beam} [GeV];N_{events}",
            DELTAE_NBINS, DELTAE_MIN, DELTAE_MAX
        )
        h.SetDirectory(0)  # évite que l'histo soit géré par ROOT globalement

        # Remplit l’histogramme avec les ΔE du bin
        for v in dE:
            h.Fill(float(v))

        # Ajuste un gaussien silencieusement ("Q0" = quiet + pas de sortie graphique)
        fit_status = h.Fit("gaus", "Q0")

        # Récupère la fonction de fit
        f = h.GetFunction("gaus")
        if f:
            sigma = float(f.GetParameter(2))   # σ du fit
            esig  = float(f.GetParError(2))    # erreur sur σ
        else:
            sigma = 0.0
            esig  = 0.0

        # Normalise par l’énergie centrale du bin (évite division par 0)
        Ec = float(centers[i]) if centers[i] > 0 else 1.0
        sig_pct[i]  = 100.0 * (sigma / Ec)   # σ/E (%)
        esig_pct[i] = 100.0 * (esig  / Ec)   # δσ/E (%)

    return centers, sig_pct, esig_pct


# ------------------------------- PLOTS ROOT -------------------------------
def plot_linearity_multi(results: Mapping[str, Tuple[np.ndarray, np.ndarray]], out_png: Path):
    """
    Trace les profils de linéarité <E_reco> vs E_beam pour plusieurs échantillons
    (ex. 'pi', 'proton') sur la même figure et sauvegarde en PNG.

    Paramètres
    ----------
    results : Mapping[str, Tuple[np.ndarray, np.ndarray]]
        Dictionnaire {label: (y_true, y_pred)} où y_true=E_beam, y_pred=E_reco.
    out_png : Path
        Chemin du fichier image de sortie (.png).
    """

    # Crée le dossier de sortie si nécessaire
    _ensure_dir(out_png)

    # Binning fixe (bornes) pour profiler <E_reco> vs E_beam
    bins = _bin_edges_fixed()

    # "keep" retient des pointeurs ROOT pour éviter la collecte/fermeture prématurée
    keep = []

    # Cadre/axes : X = E_beam, Y = <E_reco>
    frame = TH1F("frameLin", ";E_{beam} [GeV];<E_{reco}> [GeV]", 100, 0.0, 100.0)
    frame.SetMinimum(0.0)
    frame.SetMaximum(100.0)

    # Canvas principal
    c = TCanvas("cLinAll", "Linearity profile (pi-,p) ", 900, 650)
    frame.Draw()
    keep += [frame, c]

    # Diagonale de parfaite linéarité : <E_reco> = E_beam
    diag = TLine(0.0, 0.0, 100.0, 100.0)
    diag.SetLineColor(ROOT.kGray + 2)
    diag.SetLineStyle(2)
    diag.Draw("SAME")
    keep.append(diag)

    # Légende
    leg = TLegend(0.55, 0.15, 0.88, 0.32)
    keep.append(leg)

    # Pour chaque échantillon (label), calcule le profil binned et trace avec barres d'erreur
    for label, (yt, yp) in results.items():
        # Profil : moyenne et erreur-type de E_reco dans des bins de E_beam
        x, mean, sem = _binned_profile(yt, yp, bins)

        # Convertit en TGraphErrors (SEM en Y, pas d'erreur en X)
        g = _as_graph(x, mean, None, sem, f"gLin_{label}")

        # Style (couleur et marqueur définis par dictionnaires COL/MRK)
        _style_graph(g, COL[label], MRK[label])

        # Dessin des points par-dessus le cadre
        g.Draw("P SAME")

        # Ajout à la légende avec un symbole ligne+point
        leg.AddEntry(g, LEG[label], "lp")

        keep.append(g)

    # Dessine la légende, force le rafraîchissement et sauvegarde
    leg.Draw()
    c.Update()
    c.SaveAs(str(out_png))


def plot_deviation_multi(results: Mapping[str, Tuple[np.ndarray, np.ndarray]], out_png: Path):
    """
    Trace, pour plusieurs échantillons (ex. 'pi', 'proton'), le biais relatif moyen
    en % : 100 * (⟨E_reco⟩ - E_ref) / E_ref, où E_ref = centre du bin d'E_true.
    Sauvegarde la figure en PNG.

    Paramètres
    ----------
    results : Mapping[str, Tuple[np.ndarray, np.ndarray]]
        Dictionnaire {label: (y_true, y_pred)} avec y_true=E_beam, y_pred=E_reco.
    out_png : Path
        Chemin du fichier PNG de sortie.
    """

    # Assure l’existence du dossier cible
    _ensure_dir(out_png)

    # Bornes de binning fixes
    bins = _bin_edges_fixed()

    # Conteneur pour garder des références ROOT en vie
    keep = []

    # Cadre/axes : X = E_beam, Y = biais relatif (%)
    frame = TH1F("frameDevPct", ";E_{beam} [GeV];Relative bias [%]", 100, 0.0, 100.0)
    frame.SetMinimum(-10.0)
    frame.SetMaximum(10.0)

    # Canvas
    c = TCanvas("cDevAll", "Relative deviation (%) (pi-,p) ", 900, 650)
    frame.Draw()
    keep += [frame, c]

    # Bande ±1% pour visualiser une zone de biais "acceptable"
    band = TBox(0.0, -1.0, 100.0, 1.0)
    band.SetFillColorAlpha(ROOT.kGray, 0.2)  # remplissage gris semi-transparent
    band.SetLineColor(0)                     # pas de bord visible
    band.Draw("SAME")
    keep.append(band)

    # Ligne horizontale à 0% (biais nul)
    zero = TLine(0.0, 0.0, 100.0, 0.0)
    zero.SetLineColor(ROOT.kGray + 2)
    zero.SetLineStyle(2)
    zero.Draw("SAME")
    keep.append(zero)

    # Légende
    leg = TLegend(0.60, 0.70, 0.88, 0.88)
    keep.append(leg)

    # Pour chaque échantillon, calcule le biais binned et trace avec barres d’erreur
    for label, (yt, yp) in results.items():
        # Dev (%) et erreur (%) par bin d'E_true
        x, dev, err = _binned_deviation_percent(yt, yp, bins)

        # Graphique avec erreurs seulement en Y
        g = _as_graph(x, dev, None, err, f"gDev_{label}")

        # Style (couleur et marqueur issus de dictionnaires globaux COL/MRK)
        _style_graph(g, COL[label], MRK[label])

        # Trace ligne + points
        g.Draw("LP SAME")

        # Ajoute l’entrée dans la légende
        leg.AddEntry(g, LEG[label], "lp")

        keep.append(g)

    # Affiche la légende, rafraîchit et sauvegarde
    leg.Draw()
    c.Update()
    c.SaveAs(str(out_png))


def plot_resolution_multi(results: Mapping[str, Tuple[np.ndarray, np.ndarray]], out_png: Path):
    """
    Trace la résolution énergétique (σ/E en %) en fonction de l’énergie faisceau
    pour plusieurs échantillons (ex. 'pi', 'proton'), puis sauvegarde la figure en PNG.

    Paramètres
    ----------
    results : Mapping[str, Tuple[np.ndarray, np.ndarray]]
        Dictionnaire {label: (y_true, y_pred)} où :
            y_true = énergie vraie (E_beam)
            y_pred = énergie reconstruite (E_reco)
    out_png : Path
        Chemin du fichier PNG de sortie.
    """

    # Crée le dossier de sortie si besoin
    _ensure_dir(out_png)

    # Bornes fixes des bins d’énergie
    bins = _bin_edges_fixed()

    # Stockage pour éviter que ROOT libère les objets
    keep = []

    # Histogramme-cadre (axes, limites, labels)
    frame = TH1F("frameResPct", ";E_{beam} [GeV];#sigma/E_{reco} [%]", 100, 0.0, 100.0)
    frame.SetMinimum(0.0)
    frame.SetMaximum(20.0)

    # Canvas principal
    c = TCanvas("cResAll", "sigma/E (%) vs E (pi-,p)", 900, 650)
    frame.Draw()
    keep += [frame, c]

    # Lignes horizontales de référence à 5% et 10%
    l5  = TLine(0.0,  5.0, 100.0,  5.0)
    l10 = TLine(0.0, 10.0, 100.0, 10.0)
    for ln in (l5, l10):
        ln.SetLineColor(ROOT.kBlack)
        ln.SetLineStyle(2)
        ln.Draw("SAME")
    keep += [l5, l10]

    # Légende
    leg = TLegend(0.60, 0.70, 0.88, 0.88)
    keep.append(leg)

    # Pour chaque échantillon (particule)
    for label, (yt, yp) in results.items():
        # Calcule σ/E (%) et son erreur par bin (fit gaussien sur ΔE)
        x, sig_pct, esig_pct = _binned_resolution_gaussfit_percent(yt, yp, bins)

        # Convertit en TGraphErrors (σ/E vs E_beam avec barres d’erreurs)
        g = _as_graph(x, sig_pct, None, esig_pct, f"gRes_{label}")

        # Applique style (couleur, forme du marqueur)
        _style_graph(g, COL[label], MRK[label])

        # Dessine ligne + points
        g.Draw("LP SAME")

        # Ajoute à la légende
        leg.AddEntry(g, LEG[label], "lp")

        keep.append(g)

    # Finalise : légende, refresh et sauvegarde
    leg.Draw()
    c.Update()
    c.SaveAs(str(out_png))


def plot_linearity_and_deviation_combo(results: Mapping[str, Tuple[np.ndarray, np.ndarray]], out_png: Path):
    """
    Trace une figure combinée :
      - Haut : profil de linéarité <E_reco> vs E_beam
      - Bas  : biais relatif (%) vs E_beam
    et sauvegarde en PNG.

    Paramètres
    ----------
    results : Mapping[str, Tuple[np.ndarray, np.ndarray]]
        Dictionnaire {label: (y_true, y_pred)} avec :
            y_true = E_beam (GeV), y_pred = E_reco (GeV)
    out_png : Path
        Chemin du fichier PNG de sortie.
    """

    # Crée le dossier de sortie si nécessaire
    _ensure_dir(out_png)

    # Bins d'énergie identiques pour les deux sous-panneaux
    bins = _bin_edges_fixed()

    # Liste pour conserver des références sur les objets ROOT (évite leur destruction)
    keep = []

    # Canvas carré avec deux pads (haut/bas)
    c = TCanvas("cComb", "Linearity + Deviation", 900, 900)
    keep.append(c)

    # Définition des pads : pad1 (haut), pad2 (bas)
    pad1 = TPad("pad1", "", 0.0, 0.35, 1.0, 1.0)  # occupe 65% supérieurs
    pad2 = TPad("pad2", "", 0.0, 0.0,  1.0, 0.35) # occupe 35% inférieurs

    # Marges : pas d'axe X en haut, plus d'espace pour labels en bas
    pad1.SetBottomMargin(0.0)
    pad2.SetTopMargin(0.0)
    pad2.SetBottomMargin(0.25)

    # Dessine et mémorise
    pad1.Draw(); pad2.Draw()
    keep += [pad1, pad2]

    # ---------- Panneau haut : linéarité ----------
    pad1.cd()

    # Cadre/axes du panneau haut
    fr1 = TH1F("fr1", ";E_{beam} [GeV];<E_{reco}> [GeV]", 100, 0.0, 100.0)
    fr1.SetMinimum(0.0)
    fr1.SetMaximum(100.0)
    fr1.Draw()
    keep.append(fr1)

    # Diagonale de linéarité parfaite <E_reco>=E_beam
    diag = TLine(0.0, 0.0, 100.0, 100.0)
    diag.SetLineColor(ROOT.kGray+2)
    diag.SetLineStyle(2)
    diag.Draw("SAME")
    keep.append(diag)

    # Légende du panneau haut
    legTop = TLegend(0.65, 0.15, 0.88, 0.32)
    keep.append(legTop)

    # Trace, pour chaque échantillon, le profil binned (moyenne + SEM)
    for label, (yt, yp) in results.items():
        x, mean, sem = _binned_profile(yt, yp, bins)               # <E_reco> et SEM par bin d'E_beam
        g = _as_graph(x, mean, None, sem, f"gLinTop_{label}")      # erreurs uniquement en Y
        _style_graph(g, COL[label], MRK[label])                     # style standard (couleur, marqueur)
        g.Draw("P SAME")                                            # points superposés au cadre
        legTop.AddEntry(g, LEG[label], "lp")
        keep.append(g)

    legTop.Draw()

    # ---------- Panneau bas : biais relatif ----------
    pad2.cd()
    pad2.SetGridy(True)  # quadrillage horizontal pour lire les % plus facilement

    # Cadre/axes avec réglages de tailles pour labels/titres plus lisibles
    fr2 = TH1F("fr2", ";E_{beam} [GeV];Relative bias [%]", 100, 0.0, 100.0)
    fr2.SetMinimum(-11.0)
    fr2.SetMaximum(11.0)
    fr2.GetYaxis().SetNdivisions(505)
    fr2.GetYaxis().SetLabelSize(0.08)
    fr2.GetYaxis().SetTitleSize(0.09)
    fr2.GetYaxis().SetTitleOffset(0.6)
    fr2.GetXaxis().SetLabelSize(0.08)
    fr2.GetXaxis().SetTitleSize(0.09)
    fr2.GetXaxis().SetTitleOffset(0.6)
    fr2.Draw()
    keep.append(fr2)

    # Bande ±1% (zone de tolérance) et ligne 0%
    band = TBox(0.0, -1.0, 100.0, 1.0)
    band.SetFillColorAlpha(ROOT.kGray, 0.15)
    band.SetLineColor(0)
    band.Draw("SAME")
    keep.append(band)

    zero = TLine(0.0, 0.0, 100.0, 0.0)
    zero.SetLineColor(ROOT.kGray+2)
    zero.SetLineStyle(2)
    zero.Draw("SAME")
    keep.append(zero)

    # Trace, pour chaque échantillon, le biais relatif (%) + erreur
    for label, (yt, yp) in results.items():
        x, dev, err = _binned_deviation_percent(yt, yp, bins)
        g = _as_graph(x, dev, None, err, f"gDevBot_{label}")
        _style_graph(g, COL[label], MRK[label])
        g.Draw("LP SAME")
        keep.append(g)

    # Finalise et exporte
    c.Update()
    c.SaveAs(str(out_png))


def extract_metrics_at_energies(results, energies=(20.0, 80.0)):
    """
    Extrait les performances (biais relatif et résolution) à des énergies données.

    Pour chaque espèce de particule (pi, proton, ...), la fonction calcule :
      - le biais relatif moyen (%)  = 100 * (⟨E_reco⟩ - E_ref) / E_ref
      - l’incertitude sur le biais (%)
      - la résolution énergétique   = σ/E en %
      - l’incertitude sur la résolution

    Ces métriques sont évaluées aux bin centers correspondant aux énergies
    spécifiées (ex. 20, 80 GeV).  
    Résultat : liste de dictionnaires → facile à convertir en CSV ou tableau.

    Paramètres
    ----------
    results : dict
        Dictionnaire {label: (y_true, y_pred)}, où :
            - y_true : énergies vraies (E_beam, GeV)
            - y_pred : énergies reconstruites (E_reco, GeV)
    energies : tuple[float], défaut=(20.0, 80.0)
        Énergies aux centres desquelles extraire les métriques.

    Retour
    ------
    rows : list[dict]
        Liste de dictionnaires avec les clés :
        ["particle", "E_GeV", "bias_rel_pct", "bias_err_pct",
         "resolution_pct", "resolution_err_pct"]
    """

    # Définition des bins fixes (communs à tous les calculs)
    bins = _bin_edges_fixed()
    rows = []

    for label, (yt, yp) in results.items():
        # --- Biais relatif en % (avec erreur statistique) ---
        x_dev, dev_pct, dev_err = _binned_deviation_percent(yt, yp, bins)

        # --- Résolution σ/E (%) (via fit gaussien de ΔE) ---
        x_res, res_pct, res_err = _binned_resolution_gaussfit_percent(yt, yp, bins)

        for E in energies:
            # Recherche l’index du bin center correspondant à E
            if E in x_dev:
                # Cas exact (quand E correspond pile à un centre de bin, ex. 20, 80 GeV)
                i = int(np.where(np.isclose(x_dev, E))[0][0])
            else:
                # Sinon, on prend le bin center le plus proche
                i = int(np.argmin(np.abs(x_dev - E)))

            # Ajoute les métriques sous forme de dict
            rows.append({
                "particle": label,                  # type de particule
                "E_GeV": float(x_dev[i]),           # énergie centrale du bin
                "bias_rel_pct": float(dev_pct[i]),  # biais relatif en %
                "bias_err_pct": float(dev_err[i]),  # erreur sur le biais en %
                "resolution_pct": float(res_pct[i]),      # σ/E (%)
                "resolution_err_pct": float(res_err[i]),  # erreur sur σ/E (%)
            })

    return rows


# ------------------------------ Main ------------------------------
if __name__ == "__main__":

    # ==============================
    # 0) Paramètres généraux / I/O
    # ==============================
    OUT_DIR = Path("plots")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ==============================
    # 1) Choix du fichier CSV
    # ==============================

    # CSV_PATH = Path("BDT_BDT.csv")
    # CSV_PATH = Path("BDT_chi2.csv")
    CSV_PATH = Path("no_pid_BDT.csv")
    # CSV_PATH = Path("no_pid_chi2.csv")
    # CSV_PATH = Path("no_pid_chi2_pion_all.csv")
    # CSV_PATH = Path("no_pid_BDT_pi_all.csv")  

    # ==============================
    # 2) Chargement & groupement
    # ==============================
    # results: {"pi": (y_true, y_pred), "pi-": (...), "proton": (...)}
    results = load_csv_grouped(CSV_PATH)

    # =========================================
    # 3) Extraction des métriques à 20 et 80 GeV
    # =========================================
    rows = extract_metrics_at_energies(results, energies=(20.0, 80.0))

    # Impression console
    print("\n=== Biais relatif et résolution à 20 et 80 GeV ===")
    for r in rows:
        print(
            f"[{LEG[r['particle']]}] E={r['E_GeV']:.0f} GeV | "
            f"biais={r['bias_rel_pct']:+.2f} ± {r['bias_err_pct']:.2f} % | "
            f"résolution={r['resolution_pct']:.2f} ± {r['resolution_err_pct']:.2f} %"
        )

    # ------------------------------------------
    # Optionnel : sauvegarde CSV des métriques
    # ------------------------------------------
    # pd.DataFrame(rows).to_csv(OUT_DIR / "metrics_20_80.csv", index=False)

    # ==============================
    # 4) Tracés (toutes les variantes)
    # ==============================

    # -------- PID + BDT --------
    # plot_resolution_multi(results, OUT_DIR / "PID_Resolution_relative_BDT.png")
    # plot_linearity_and_deviation_combo(results, OUT_DIR / "PID_Lin_n_Dev_BDT.png")
    # plot_linearity_multi(results, OUT_DIR / "PID_Lin_profile_BDT.png")
    # plot_deviation_multi(results, OUT_DIR / "PID_Relative_deviation_BDT.png")

    # -------- PID + chi2 --------
    # plot_resolution_multi(results, OUT_DIR / "PID_Resolution_relative_chi2.png")
    # plot_linearity_and_deviation_combo(results, OUT_DIR / "PID_Lin_n_Dev_chi2.png")
    # plot_linearity_multi(results, OUT_DIR / "PID_Lin_profile_chi2.png")
    # plot_deviation_multi(results, OUT_DIR / "PID_Relative_deviation_chi2.png")

    # -------- no PID + BDT --------
    plot_resolution_multi(results, OUT_DIR / "no_PID_Resolution_relative_BDT.png")
    plot_linearity_and_deviation_combo(results, OUT_DIR / "no_PID_Lin_n_Dev_BDT.png")
    # plot_linearity_multi(results, OUT_DIR / "no_PID_Lin_profile_BDT.png")
    # plot_deviation_multi(results, OUT_DIR / "no_PID_Relative_deviation_BDT.png")

    # -------- no PID + chi2 --------
    # plot_resolution_multi(results, OUT_DIR / "no_PID_Resolution_relative_chi2.png")
    # plot_linearity_and_deviation_combo(results, OUT_DIR / "no_PID_Lin_n_Dev_chi2.png")
    # plot_linearity_multi(results, OUT_DIR / "no_PID_Lin_profile_chi2.png")
    # plot_deviation_multi(results, OUT_DIR / "no_PID_Relative_deviation_chi2.png")

    # -------- no PID + chi2 (pions all) --------
    # plot_resolution_multi(results, OUT_DIR / "no_PID_Resolution_relative_chi2_pion.png")
    # plot_linearity_and_deviation_combo(results, OUT_DIR / "no_PID_Lin_n_Dev_chi2_pion.png")
    # plot_linearity_multi(results, OUT_DIR / "PID_Lin_profile_chi2_pion.png")
    # plot_deviation_multi(results, OUT_DIR / "PID_Relative_deviation_chi2_pion.png")

    # -------- no PID + BDT (pions all) --------
    # plot_resolution_multi(results, OUT_DIR / "no_PID_Resolution_relative_BDT_pion.png")
    # plot_linearity_and_deviation_combo(results, OUT_DIR / "no_PID_Lin_n_Dev_BDT_pion.png")
