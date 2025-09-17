/*
  EnergyReco.C
  Reconstruction de l’énergie primaire du faisceau pour π⁻, K⁰_L et p
  à partir des comptages de seuils (N1, N2, N3), via un ajustement χ²
  avec TMinuit sur des coefficients quadratiques dépendant du nombre total de hits.

  Principe :
    - Entrées ROOT :
        * digitized: 130k_<particle>_E1to130_digitized.root  (TTree "tree")
            - branches: thr (std::vector<int>), K (std::vector<int>)
        * raw:       130k_<particle>_E1to130.root            (TTree "tree")
            - branch: primaryEnergy (double)
    - Pré-traitement :
        * exclusion des événements touchant la dernière couche (K == 47)
        * calcul N1, N2, N3 depuis thr ∈ {1,2,3}
    - Modèle d’énergie reconstruite :
        Nhit = N1+N2+N3
        α(Nhit) = a0 + a1*Nhit + a2*Nhit²
        β(Nhit) = b0 + b1*Nhit + b2*Nhit²
        γ(Nhit) = c0 + c1*Nhit + c2*Nhit²
        Ereco   = α*N1 + β*N2 + γ*N3
    - Critère minimisé :
        χ² = Σ_i (Ereco_i - Etrue_i)² / max(1e-9, Etrue_i)
      (implémenté dans chi2Function et passé à TMinuit::Migrad)
    - Amorces (startParams) spécifiques à la particule (π, K, p).

  Sorties (dans ./plots) :
    - Reco_vs_Beam_<particle>.png        : comparatif distributions E_beam et E_reco
    - Profil_linearite_<particle>.png    : TProfile <E_reco> vs E_true avec diagonale idéale
    - Linearity_<particle>.png           : scatter E_reco vs E_true + y=x
    - Resolution_relative_<particle>.png : σ(E_reco−E_true)/E_true obtenu par fit gaussien
                                           de ΔE par tranches d’énergie vraie (20 bins, 0–100 GeV)

  Utilisation (depuis ROOT) :
    root -l -q 'EnergyReco.C("pi")'
    root -l -q 'EnergyReco.C("kaon")'
    root -l -q 'EnergyReco.C("proton")'

  Remarques :
    - Les steps et bornes des paramètres Minuit sont réglés finement (notamment c0).
    - Un avertissement est affiché si le nombre d’entrées diffère entre les deux fichiers.
    - Les événements avec E_true > 100 GeV sont exclus des tracés de linéarité pour rester
      aligné avec l’échelle des figures.
*/

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "TMath.h"
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TLegend.h"
#include "TProfile.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TLine.h"
#include "TF1.h"
#include "TMinuit.h"

// Données globales pour la minimisation
std::vector<int>    gN1, gN2, gN3;
std::vector<double> gEbeam;

void chi2Function(int& npar, double* grad, double& fval, double* par, int flag) {
    const double bigPenalty = 1e30;
    double chi2 = 0.0;
    for (size_t i = 0; i < gN1.size(); ++i) {
        if (gEbeam[i] <= 0.0) continue;
        const int Nhit = gN1[i] + gN2[i] + gN3[i];
        const double alpha = par[0] + par[1]*Nhit + par[2]*Nhit*Nhit;
        const double beta  = par[3] + par[4]*Nhit + par[5]*Nhit*Nhit;
        const double gamma = par[6] + par[7]*Nhit + par[8]*Nhit*Nhit;
        const double Ereco = alpha*gN1[i] + beta*gN2[i] + gamma*gN3[i];
        const double diff = Ereco - gEbeam[i];
        chi2 += diff*diff / std::max(1e-9, gEbeam[i]);
    }
    if (!std::isfinite(chi2)) chi2 = bigPenalty;
    fval = chi2;
}

void EnergyReco(const std::string& particle) {
    // --- Sécurité: reset des buffers globaux
    gN1.clear(); gN2.clear(); gN3.clear();
    gEbeam.clear();

    // --- Fichiers d'entrée
    const std::string fHitsName   = "/gridgroup/ilc/midir/analyse/data/digitized/130k_" + particle + "_E1to130_digitized.root";
    const std::string fEnergyName = "/gridgroup/ilc/midir/analyse/data/raw/130k_"      + particle + "_E1to130.root";

    TFile* fHits   = TFile::Open(fHitsName.c_str());
    TFile* fEnergy = TFile::Open(fEnergyName.c_str());
    if (!fHits || !fEnergy) { std::cerr << "Erreur ouverture fichiers pour " << particle << "\n"; return; }

    TTree* treeHits   = static_cast<TTree*>(fHits->Get("tree"));
    TTree* treeEnergy = static_cast<TTree*>(fEnergy->Get("tree"));
    if (!treeHits || !treeEnergy) { std::cerr << "Arbre manquant.\n"; return; }

    if (treeHits->GetEntries() != treeEnergy->GetEntries())
        std::cerr << "Avertissement: nombres d'entrées différents: hits=" << treeHits->GetEntries()
                  << " energy=" << treeEnergy->GetEntries() << "\n";

    // --- Branches
    std::vector<int>* vThr = nullptr;
    std::vector<int>* vK   = nullptr;
    treeHits->SetBranchAddress("thr", &vThr);
    treeHits->SetBranchAddress("K",   &vK);

    double primenergy = 0.0;
    treeEnergy->SetBranchAddress("primaryEnergy",   &primenergy);

    // --- Remplissage
    const int kLastLayer = 47;
    int nLastLayer = 0;
    const Long64_t nEntries = treeHits->GetEntries();

    for (Long64_t i = 0; i < nEntries; ++i) {
        treeHits->GetEntry(i);
        treeEnergy->GetEntry(i);

        bool hasLastLayerHit = false;
        for (size_t h = 0; h < vK->size(); ++h) { if ((*vK)[h] == kLastLayer) { hasLastLayerHit = true; break; } }
        if (hasLastLayerHit) { ++nLastLayer; continue; }

        int N1 = 0, N2 = 0, N3 = 0;
        for (size_t h = 0; h < vThr->size(); ++h) {
            const int t = (*vThr)[h];
            if      (t == 1) ++N1;
            else if (t == 2) ++N2;
            else if (t == 3) ++N3;
        }

        if (primenergy > 0.0) {
            gN1.push_back(N1); gN2.push_back(N2); gN3.push_back(N3);
            gEbeam.push_back(primenergy);
        }
    }

    // --- Paramètres initiaux dépendant de la particule
    double startParams[9];
    bool ok = true;
    if (particle == "kaon") {
        startParams[0]=0.0560298; startParams[1]=-4.58223e-05; startParams[2]=1.90941e-08;
        startParams[3]=0.0800358; startParams[4]=-7.76142e-05; startParams[5]=4.5824e-08;
        startParams[6]=1.22381e-14; startParams[7]=7.97643e-04; startParams[8]=-3.5693e-07;
    } else if (particle == "pi") {
        startParams[0]=0.0430851; startParams[1]=-3.50665e-05; startParams[2]=1.94847e-08;
        startParams[3]=0.107201;  startParams[4]=-6.36402e-05; startParams[5]=1.20235e-08;
        startParams[6]=1.81865e-13; startParams[7]=0.000735489; startParams[8]=-3.00925e-07;
    } else if (particle == "proton") {
        startParams[0]=0.0435267; startParams[1]=-1.89745e-05; startParams[2]=1.07039e-08;
        startParams[3]=0.122286;  startParams[4]=-4.93952e-05; startParams[5]=5.45074e-09;
        startParams[6]=2.61942e-13; startParams[7]=0.00047743; startParams[8]=-1.68624e-07;
    } else {
        std::cerr << "Particle inconnue: " << particle << "\n"; ok = false;
    }
    if (!ok || gN1.empty()) { std::cerr << "Pas de données exploitables.\n"; return; }

    // --- Minuit
    const char* parNames[9] = { "a0","a1","a2", "b0","b1","b2", "c0","c1","c2" };
    TMinuit* minuit = new TMinuit(9);
    minuit->SetFCN(chi2Function);

    // for (int i=0; i<9; ++i) {
    //     double lo = 0.0, hi = 0.0, step = 1e-5;
    //     if (i==6) lo = 1e-10, step = 1e-15; // comme dans ton code
    //     minuit->DefineParameter(i, parNames[i], startParams[i], step, lo, hi);
    // }

    // --- paramètres initiaux et steps ---
    // Paramètre a0
    minuit->DefineParameter(0, parNames[0], /*start=*/startParams[0], /*step=*/1e-5, /*min=*/0.0, /*max=*/0.0);
    // Paramètre a1
    minuit->DefineParameter(1, parNames[1], /*start=*/startParams[1], /*step=*/1e-6,  /*min=*/0.0, /*max=*/0.0);
    // Paramètre a2
    minuit->DefineParameter(2, parNames[2], /*start=*/startParams[2], /*step=*/1e-9,  /*min=*/0.0, /*max=*/0.0);

    // Paramètre b0
    minuit->DefineParameter(3, parNames[3],  /*start=*/startParams[3], /*step=*/1e-5,  /*min=*/0.0, /*max=*/0.0);  
    // Paramètre b1
    minuit->DefineParameter(4, parNames[4],  /*start=*/startParams[4], /*step=*/1e-6,  /*min=*/0.0, /*max=*/0.0);
    // Paramètre b2
    minuit->DefineParameter(5, parNames[5],  /*start=*/startParams[5], /*step=*/1e-5,  /*min=*/0.0, /*max=*/0.0);

    // Paramètre c0
    minuit->DefineParameter(6, parNames[6],  /*start=*/startParams[6], /*step=*/1e-15,   /*min=*/0.0, /*max=*/1e-10);  
    // Paramètre c1
    minuit->DefineParameter(7, parNames[7],  /*start=*/startParams[7], /*step=*/1e-6,  /*min=*/0.0, /*max=*/0.0);
    // Paramètre c2
    minuit->DefineParameter(8, parNames[8],  /*start=*/startParams[8], /*step=*/1e-5, /*min=*/0.0, /*max=*/0.0);
    minuit->Migrad();

    Double_t fmin=0, fedm=0, errdef=0; Int_t nvpar=0, nparx=0, istat=0;
    minuit->mnstat(fmin, fedm, errdef, nvpar, nparx, istat);
    const int nData = static_cast<int>(gN1.size());
    const int ndf   = nData - nvpar;
    std::cout << "=== Résultats minimisation ("<< particle <<") ===\n"
              << "chi2_min = " << fmin << "  ndf = " << ndf
              << "  chi2/ndf = " << (ndf>0 ? fmin/ndf : -1) << "\n"
              << "Evts filtrés (couche 47): " << nLastLayer << "\n";

    double par[9], err[9];
    for (int ip=0; ip<9; ++ip) minuit->GetParameter(ip, par[ip], err[ip]);

    // --- Plots (suffixés par particule)
    const std::string tag = "_" + particle;

    TH1D* hEreco = new TH1D(("hEreco"+tag).c_str(), "E_{reco} ;E_{reco} [GeV];N_{events}", 100, 0, 100);
    TH1D* hEbeam = new TH1D(("hEbeam"+tag).c_str(), "E beam vs E reco;E [GeV];N_{events}", 100, 0, 100);
    hEreco->SetDirectory(0); hEbeam->SetDirectory(0);
    TProfile* pProf = new TProfile(("pProf"+tag).c_str(), "Linearity profile ;E_{beam true} [GeV];<E_{reco}> [GeV]", 100, 0, 100);

    std::vector<double> x_true, y_reco; x_true.reserve(gN1.size()); y_reco.reserve(gN1.size());
    for (size_t i = 0; i < gN1.size(); ++i) {
        if (gEbeam[i] > 100.0) continue;
        const int    Nhit  = gN1[i] + gN2[i] + gN3[i];
        const double alpha = par[0] + par[1]*Nhit + par[2]*Nhit*Nhit;
        const double beta  = par[3] + par[4]*Nhit + par[5]*Nhit*Nhit;
        const double gamma = par[6] + par[7]*Nhit + par[8]*Nhit*Nhit;
        const double Ereco = alpha*gN1[i] + beta*gN2[i] + gamma*gN3[i];

        hEreco->Fill(Ereco);
        hEbeam->Fill(gEbeam[i]);
        pProf->Fill(gEbeam[i], Ereco);
        x_true.push_back(gEbeam[i]);
        y_reco.push_back(Ereco);
    }

    TCanvas* cComp = new TCanvas(("cComp"+tag).c_str(), "Reco vs Beam", 800, 600);
    hEbeam->SetStats(kFALSE);
    hEbeam->SetLineColor(kBlue);
    hEreco->SetLineColor(kRed);
    hEbeam->Draw("HIST");
    hEreco->Draw("HIST SAME");
    TLegend* leg = new TLegend(0.45,0.75,0.68,0.88);
    leg->AddEntry(hEbeam,"E Beam","l");
    leg->AddEntry(hEreco,"E reco","l");
    leg->Draw();
    cComp->SaveAs(("plots/Reco_vs_Beam" + tag + ".png").c_str());

    TCanvas* c1 = new TCanvas(("c1"+tag).c_str(), "Linearity profile", 800, 600);
    pProf->SetStats(kFALSE);
    pProf->GetYaxis()->SetRangeUser(0, 100);
    pProf->Draw();
    TLine* l1 = new TLine(0,0,100,100); l1->SetLineColor(kRed); l1->SetLineStyle(2); l1->Draw();
    c1->SaveAs(("plots/Profil_linearite" + tag + ".png").c_str());

    TCanvas* c2 = new TCanvas(("c2"+tag).c_str(), "Linearity", 800, 600);
    TGraph* g2 = new TGraph((int)x_true.size(), &x_true[0], &y_reco[0]);
    g2->SetTitle("E reco vs E Beam ;E_{beam} [GeV];E_{reco} [GeV]");
    g2->GetYaxis()->SetRangeUser(0, 100);
    g2->Draw("AP");
    TLine* l2 = new TLine(0,0,100,100); l2->SetLineColor(kRed); l2->SetLineStyle(2); l2->SetLineWidth(3); l2->Draw();
    c2->SaveAs(("plots/Linearity" + tag + ".png").c_str());

    // --- Résolution
    std::vector<double> Etrue_all, Ereco_all;
    Etrue_all.reserve(gN1.size()); Ereco_all.reserve(gN1.size());
    for (size_t i = 0; i < gN1.size(); ++i) {
        const int    Nhit  = gN1[i] + gN2[i] + gN3[i];
        const double alpha = par[0] + par[1]*Nhit + par[2]*Nhit*Nhit;
        const double beta  = par[3] + par[4]*Nhit + par[5]*Nhit*Nhit;
        const double gamma = par[6] + par[7]*Nhit + par[8]*Nhit*Nhit;
        const double Ereco = alpha*gN1[i] + beta*gN2[i] + gamma*gN3[i];
        Etrue_all.push_back(gEbeam[i]); Ereco_all.push_back(Ereco);
    }

    const int    nBins = 20;
    const double Emin  = 0.0;
    const double Emax  = 100.0;
    const double dE    = (Emax - Emin)/nBins;

    std::vector<double> binCenter(nBins), sigma(nBins), sigmaErr(nBins);
    for (int ib = 0; ib < nBins; ++ib) {
        binCenter[ib] = Emin + (ib + 0.5)*dE;
        const double eLow  = Emin + ib*dE;
        const double eHigh = eLow + dE;

        TH1D hde(("hde"+tag+std::to_string(ib)).c_str(), "ΔE;E_{reco}-E_{true} [GeV];N_{events}", 100, -20, 20);
        for (size_t i = 0; i < Etrue_all.size(); ++i)
            if (Etrue_all[i] >= eLow && Etrue_all[i] < eHigh)
                hde.Fill(Ereco_all[i] - Etrue_all[i]);

        hde.Fit("gaus","Q");
        TF1* f = hde.GetFunction("gaus");
        sigma   [ib] = f ? f->GetParameter(2) : 0.0;
        sigmaErr[ib] = f ? f->GetParError(2) : 0.0;
    }

    std::vector<double> relSigma(nBins), relErr(nBins);
    for (int ib = 0; ib < nBins; ++ib) {
        relSigma[ib] = (binCenter[ib]>0 ? sigma[ib] / binCenter[ib] : 0.0);
        relErr  [ib] = (binCenter[ib]>0 ? sigmaErr[ib] / binCenter[ib] : 0.0);
    }

    TGraphErrors* gRel = new TGraphErrors(nBins, &binCenter[0], &relSigma[0], nullptr, &relErr[0]);
    gRel->SetTitle("Relative resolution;E_{true} [GeV];sigma/E");
    TCanvas* cRes = new TCanvas(("cRes"+tag).c_str(), "sigma/E vs E", 800, 600);
    gRel->Draw("APL");
    cRes->SaveAs(("plots/Resolution_relative" + tag + ".png").c_str());

    // --- Nettoyage
    delete minuit;
    fHits->Close(); fEnergy->Close();
}
