"""
Script Python pour la visualisation des performances de reconstruction d’énergie
des hadrons (π⁻, K⁰_L, p) à partir de fichiers .npz (y_true, y_pred).

Fonctionnalités principales :
- Chargement des données de test/prédictions (.npz) pour chaque particule
- Calcul binned de :
    * profil de linéarité <E_reco> vs E_true
    * biais relatif (déviation en %)
    * résolution relative σ/E (fit gaussien de ΔE)
- Production de graphes ROOT (TGraphErrors, TH1F, etc.) :
    * linearité
    * résolution
    * biais relatif
    * combinaison linéarité + biais
- Sauvegarde automatique des figures en .png

Dépendances : numpy, ROOT (PyROOT)

Usage :
- Adapter les chemins vers les fichiers .npz dans le dictionnaire NPZ
- Lancer le script pour générer les plots dans le dossier "plots"
"""




from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Mapping, Optional
import numpy as np

import ROOT
from ROOT import (
    TCanvas, TGraphErrors, TLegend, TH1F, TH1D, TLine, TBox, TPad,
    gROOT, gStyle, kRed, kBlue, kGreen
)

ROOT.TH1.AddDirectory(False)

# --------------------- --------------------- ---------------------
COL = {"pi": kRed, "kaon": kGreen+2, "proton": kBlue}
MRK = {"pi": 20,   "kaon": 21,       "proton": 22}
LEG = {"pi": "#pi^{-}", "kaon": "K_{L}^{0}", "proton": "p"}

# Binning ENERGIE fixé comme la macro : 10 -> 100 GeV, 20 bins
E_MIN_FIXED = 10.0
E_MAX_FIXED = 100.0
NBINS_FIXED = 20

# Résolution : histo de ΔE = Ereco - Etrue, fit gaussien, σ/E_centre (en %) + erreur
DELTAE_NBINS = 100
DELTAE_MIN = -20.0
DELTAE_MAX =  20.0

# -------------------------- Utils / helpers ROOT ---------------------------

def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _bin_edges_fixed(nbins: int = NBINS_FIXED,
                     emin: float = E_MIN_FIXED,
                     emax: float = E_MAX_FIXED) -> np.ndarray:
    return np.linspace(emin, emax, nbins+1)


def _as_graph(x: np.ndarray, y: np.ndarray,
              ex: Optional[np.ndarray] = None,
              ey: Optional[np.ndarray] = None,
              name: str = "g") -> TGraphErrors:
    n = len(x)
    ex = np.zeros(n) if ex is None else ex
    ey = np.zeros(n) if ey is None else ey
    g = TGraphErrors(n, x.astype(np.float64), y.astype(np.float64),
                     ex.astype(np.float64), ey.astype(np.float64))
    g.SetName(name)
    return g


def _style_graph(g: TGraphErrors, color: int, marker: int) -> None:
    g.SetLineColor(color)
    g.SetMarkerColor(color)
    g.SetMarkerStyle(marker)


# ---------------------------- Calculs binned -------------------------------

def _binned_profile(y_true: np.ndarray, y_pred: np.ndarray, bins: np.ndarray):
    centers = 0.5*(bins[1:] + bins[:-1])
    mean = np.zeros_like(centers)
    sem  = np.zeros_like(centers)

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = (y_true >= lo) & (y_true < hi)
        n = int(np.sum(m))
        if n > 0:
            v = y_pred[m]
            mu = float(v.mean())
            sd = float(v.std(ddof=0))
            mean[i] = mu
            sem[i]  = sd / np.sqrt(n)
        else:
            mean[i] = 0.0
            sem[i]  = 0.0
    return centers, mean, sem


def _binned_deviation_percent(y_true: np.ndarray, y_pred: np.ndarray, bins: np.ndarray):
    centers = 0.5*(bins[1:] + bins[:-1])
    dev = np.zeros_like(centers)
    err = np.zeros_like(centers)

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = (y_true >= lo) & (y_true < hi)
        n = int(np.sum(m))
        if n > 0:
            yhat = y_pred[m]
            mu   = float(yhat.mean())
            sd   = float(yhat.std(ddof=0))
            c    = float(centers[i]) if centers[i] > 0 else 1.0
            dev[i] = 100.0 * (mu - c) / c
            err[i] = 100.0 * (sd / np.sqrt(n)) / c
        else:
            dev[i] = 0.0
            err[i] = 0.0
    return centers, dev, err


def _binned_resolution_gaussfit_percent(y_true: np.ndarray, y_pred: np.ndarray, bins: np.ndarray):
    """
    Pour chaque bin en énergie vraie, construit ΔE = Ereco - Etrue,
    fit un gaussien silencieux, et retourne :
        x = centres (E_true bin center)
        y = 100 * sigma / E_center
        ey = 100 * esigma / E_center
    """
    centers = 0.5*(bins[1:] + bins[:-1])
    sig_pct = np.zeros_like(centers)
    esig_pct = np.zeros_like(centers)

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = (y_true >= lo) & (y_true < hi)
        if np.sum(m) == 0:
            sig_pct[i] = 0.0
            esig_pct[i] = 0.0
            continue
        dE = (y_pred[m] - y_true[m]).astype(np.float64)

        hname = f"hde_bin{i}"
        h = TH1D(hname, "DE;E_{reco}-E_{true} [GeV];N_{events}",
                 DELTAE_NBINS, DELTAE_MIN, DELTAE_MAX)

        h.SetDirectory(0) # détacher du directory courant (évite deletes implicites)
        # ROOT.SetOwnership(h, True) # PyROOT gère la durée de vie

        for v in dE:
            h.Fill(float(v))

        # Fit gaussien silencieux
        fit_status = h.Fit("gaus", "Q0")
        f = h.GetFunction("gaus")
        if f:
            sigma = float(f.GetParameter(2))
            esig  = float(f.GetParError(2))
        else:
            sigma = 0.0
            esig  = 0.0

        Ec = float(centers[i]) if centers[i] > 0 else 1.0
        sig_pct[i]  = 100.0 * (sigma / Ec)
        esig_pct[i] = 100.0 * (esig  / Ec)

        # # Optionnel : nettoyer pour éviter l’accumulation en batch
        # h.Delete()

    return centers, sig_pct, esig_pct


# ------------------------------- PLOTS -------------------------------------

def plot_linearity_multi(results: Mapping[str, Tuple[np.ndarray, np.ndarray]], out_png: Path):
    _ensure_dir(out_png)
    gROOT.SetBatch(True)
    gStyle.SetOptStat(0)

    bins = _bin_edges_fixed(NBINS_FIXED, E_MIN_FIXED, E_MAX_FIXED)

    keep = [] 

    # Frame X: 0->100, Y: 0->100 pour coller à la macro
    frame = TH1F("frameLin", ";E_{true} [GeV];<E_{reco}> [GeV]", 100, 0.0, 100.0)
    frame.SetMinimum(0.0)
    frame.SetMaximum(100.0)

    c = TCanvas("cLinAll", "Linearity profile (pi,K,p) ", 900, 650)
    frame.Draw()
    keep += [frame, c]

    # Diagonale y=x
    diag = TLine(0.0, 0.0, 100.0, 100.0)
    diag.SetLineColor(ROOT.kGray+2)
    diag.SetLineStyle(2)
    diag.Draw("SAME")
    keep.append(diag)

    leg = TLegend(0.55, 0.15, 0.88, 0.32)
    keep.append(leg)

    for label, (yt, yp) in results.items():
        x, mean, sem = _binned_profile(yt, yp, bins)
        g = _as_graph(x, mean, None, sem, f"gLin_{label}")
        _style_graph(g, COL[label], MRK[label])
        g.Draw("P SAME")
        leg.AddEntry(g, LEG[label], "lp")
        keep.append(g)
        
    leg.Draw()
    c.Update()
    c.SaveAs(str(out_png))


def plot_resolution_multi(results: Mapping[str, Tuple[np.ndarray, np.ndarray]], out_png: Path):
    _ensure_dir(out_png)
    gROOT.SetBatch(True)
    gStyle.SetOptStat(0)

    bins = _bin_edges_fixed(NBINS_FIXED, E_MIN_FIXED, E_MAX_FIXED)

    keep = []  # <<< garder des références vivantes

    frame = TH1F("frameResPct", ";E_{true} [GeV];#sigma/E [%]", 100, 0.0, 100.0)
    frame.SetMinimum(0.0)
    frame.SetMaximum(20.0)

    c = TCanvas("cResAll", "sigma/E (%) vs E (pi,K,p)", 900, 650)
    frame.Draw()
    keep += [frame, c]

    l5  = TLine(0.0,  5.0, 100.0,  5.0)
    l10 = TLine(0.0, 10.0, 100.0, 10.0)
    for ln in (l5, l10):
        ln.SetLineColor(ROOT.kBlack)
        ln.SetLineStyle(2)
        ln.Draw("SAME")
    keep += [l5, l10]

    leg = TLegend(0.60, 0.70, 0.88, 0.88)
    keep.append(leg)

    for label, (yt, yp) in results.items():
        x, sig_pct, esig_pct = _binned_resolution_gaussfit_percent(yt, yp, bins)
        g = _as_graph(x, sig_pct, None, esig_pct, f"gRes_{label}")
        _style_graph(g, COL[label], MRK[label])
        g.Draw("LP SAME")
        leg.AddEntry(g, LEG[label], "lp")
        keep.append(g)  # <<< garder le graphe

    leg.Draw()
    c.Update()  # optionnel mais propre
    c.SaveAs(str(out_png))


def plot_deviation_multi(results: Mapping[str, Tuple[np.ndarray, np.ndarray]], out_png: Path):
    _ensure_dir(out_png)
    gROOT.SetBatch(True)
    gStyle.SetOptStat(0)

    bins = _bin_edges_fixed(NBINS_FIXED, E_MIN_FIXED, E_MAX_FIXED)

    keep = [] 

    # Frame X: 0->100, Y: -10 -> 10 %
    frame = TH1F("frameDevPct", ";E_{true} [GeV];Biais relatif [%]", 100, 0.0, 100.0)
    frame.SetMinimum(-10.0)
    frame.SetMaximum( 10.0)

    c = TCanvas("cDevAll", "Relative deviation (%) (pi,K,p) ", 900, 650)
    frame.Draw()
    keep += [frame, c]

    # Bande ±1%
    band = TBox(0.0, -1.0, 100.0, 1.0)
    band.SetFillColorAlpha(ROOT.kGray, 0.2)
    band.SetLineColor(0)
    band.Draw("SAME")
    keep.append(band)

    # Ligne zéro
    zero = TLine(0.0, 0.0, 100.0, 0.0)
    zero.SetLineColor(ROOT.kGray+2)
    zero.SetLineStyle(2)
    zero.Draw("SAME")
    keep.append(zero)

    leg = TLegend(0.60, 0.70, 0.88, 0.88)
    keep.append(leg)

    for label, (yt, yp) in results.items():
        x, dev, err = _binned_deviation_percent(yt, yp, bins)
        g = _as_graph(x, dev, None, err, f"gDev_{label}")
        _style_graph(g, COL[label], MRK[label])
        g.Draw("LP SAME")
        leg.AddEntry(g, LEG[label], "lp")
        keep.append(g)

    leg.Draw()
    c.Update()
    c.SaveAs(str(out_png))


def plot_linearity_and_deviation_combo(results: Mapping[str, Tuple[np.ndarray, np.ndarray]], out_png: Path):
    _ensure_dir(out_png)
    gROOT.SetBatch(True)
    gStyle.SetOptStat(0)

    bins = _bin_edges_fixed(NBINS_FIXED, E_MIN_FIXED, E_MAX_FIXED)

    keep = []

    c = TCanvas("cComb", "Linearity + Deviation", 900, 900)
    keep.append(c)
    pad1 = TPad("pad1", "", 0.0, 0.35, 1.0, 1.0)
    pad2 = TPad("pad2", "", 0.0, 0.0,  1.0, 0.35)
    pad1.SetBottomMargin(0.0)
    pad2.SetTopMargin(0.0)
    pad2.SetBottomMargin(0.25)
    pad1.Draw(); pad2.Draw()
    keep += [pad1, pad2]

    # Haut : linéarité
    pad1.cd()
    fr1 = TH1F("fr1", ";E_{true} [GeV];<E_{reco}> [GeV]", 100, 0.0, 100.0)
    fr1.SetMinimum(0.0)
    fr1.SetMaximum(100.0)
    fr1.Draw()
    keep.append(fr1)

    diag = TLine(0.0, 0.0, 100.0, 100.0)
    diag.SetLineColor(ROOT.kGray+2)
    diag.SetLineStyle(2)
    diag.Draw("SAME")
    keep.append(diag)

    legTop = TLegend(0.65, 0.15, 0.88, 0.32)
    keep.append(legTop)

    for label, (yt, yp) in results.items():
        x, mean, sem = _binned_profile(yt, yp, bins)
        g = _as_graph(x, mean, None, sem, f"gLinTop_{label}")
        _style_graph(g, COL[label], MRK[label])
        g.Draw("P SAME")
        legTop.AddEntry(g, LEG[label], "lp")
        keep.append(g)
    legTop.Draw()

    # Bas : déviation
    pad2.cd()
    pad2.SetGridy(True)

    fr2 = TH1F("fr2", ";E_{true} [GeV];Biais relatif [%]", 100, 0.0, 100.0)
    fr2.SetMinimum(-10.0)
    fr2.SetMaximum( 10.0)
    fr2.GetYaxis().SetNdivisions(505)
    fr2.GetYaxis().SetLabelSize(0.08)
    fr2.GetYaxis().SetTitleSize(0.09)
    fr2.GetYaxis().SetTitleOffset(0.6)
    fr2.GetXaxis().SetLabelSize(0.08)
    fr2.GetXaxis().SetTitleSize(0.09)
    fr2.GetXaxis().SetTitleOffset(0.6)
    fr2.Draw()
    keep.append(fr2)

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

    for label, (yt, yp) in results.items():
        x, dev, err = _binned_deviation_percent(yt, yp, bins)
        g = _as_graph(x, dev, None, err, f"gDevBot_{label}")
        _style_graph(g, COL[label], MRK[label])
        g.Draw("LP SAME")
        keep.append(g)

    c.Update()
    c.SaveAs(str(out_png))


# -------------------------- Chargement des données -------------------------

def load_npz_map(paths: Mapping[str, Path]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for label, p in paths.items():
        d = np.load(p)
        out[label] = (d["y_true"], d["y_pred"])  # arrays 1D
    return out


# ---------------------------------- Main -----------------------------------
if __name__ == "__main__":
    # Adapter aux emplacements de tes .npz
    NPZ = {
        "pi":     Path("results_pi_energy_reco/arrays/test_and_pred_20.npz"),
        "kaon":   Path("results_kaon_energy_reco/arrays/test_and_pred_46.npz"),
        "proton": Path("results_proton_energy_reco/arrays/test_and_pred_14.npz"),
    }

    results = load_npz_map(NPZ)

    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Noms de fichiers ALIGNÉS sur la macro C++
    plot_resolution_multi(results, out_dir / "Resolution_relative_all_LGBM_timing.png")
    # plot_linearity_multi(results, out_dir / "Lin_profile_all_LGBM.png")
    # plot_deviation_multi(results, out_dir / "Relative_deviation_all_LGBM.png")
    plot_linearity_and_deviation_combo(results, out_dir / "Lin_n_Dev_all_LGBM_timing.png")
