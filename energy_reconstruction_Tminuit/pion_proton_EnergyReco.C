/*
  pion_proton_EnergyReco.C
  Reconstruction de l’énergie primaire du faisceau pour π⁻ et p
  via un ajustement χ² avec TMinuit (coeffs quadratiques dépendant de Nhit).
  Plage Etrue: [1,130] GeV
*/

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>   // std::max, min/max_element, nth_element

#include "TMath.h"
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
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

void chi2Function(int& /*npar*/, double* /*grad*/, double& fval, double* par, int /*flag*/) {
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

void pion_proton_EnergyReco(const std::string& particle) {
    if (!(particle == "pi-" || particle == "proton")) {
        std::cerr << "[Erreur] Particule non supportée: " << particle
                  << " (autorisées: \"pi-\", \"proton\")\n";
        return;
    }

    gN1.clear(); gN2.clear(); gN3.clear(); gEbeam.clear();

    const std::string fileName =
        "/gridgroup/ilc/midir/analyse/data/merged_primaryEnergy/200k_" + particle + "_1-130_params_merged.root";
    TFile* f = TFile::Open(fileName.c_str());
    if (!f || f->IsZombie()) { std::cerr << "[Erreur] Impossible d'ouvrir : " << fileName << "\n"; return; }
    TTree* t = static_cast<TTree*>(f->Get("tree"));
    if (!t) { std::cerr << "[Erreur] TTree 'tree' introuvable.\n"; f->Close(); delete f; return; }

    // (Perf) n'activer que ce qui est utilisé
    t->SetBranchStatus("*", 0);
    t->SetBranchStatus("N1", 1);
    t->SetBranchStatus("N2", 1);
    t->SetBranchStatus("N3", 1);
    t->SetBranchStatus("nK", 1);
    t->SetBranchStatus("K", 1);
    t->SetBranchStatus("primaryEnergy", 1);

    // Branches
    Int_t N1=0, N2=0, N3=0, nK=0;
    double primaryEnergy = 0.0;
    t->SetBranchAddress("N1", &N1);
    t->SetBranchAddress("N2", &N2);
    t->SetBranchAddress("N3", &N3);
    t->SetBranchAddress("nK", &nK);
    t->SetBranchAddress("primaryEnergy", &primaryEnergy);

    // Premier passage pour trouver nK_max
    Long64_t n = t->GetEntries();
    int nK_max = 0;
    for (Long64_t i=0;i<n;++i){ t->GetEntry(i); if (nK>nK_max) nK_max=nK; }

    // Buffer pour K[nK]
    std::vector<Int_t> K(nK_max>0?nK_max:1,0);
    t->SetBranchAddress("K", K.data());

    // S'assurer que le dossier 'plots' existe
    gSystem->mkdir("plots", kTRUE);

    // Remplissage des buffers (filtre couche 47 et plage énergie physique)
    int nLastLayer=0;
    for (Long64_t i=0;i<n;++i) {
        t->GetEntry(i);
        bool last=false; for (int h=0; h<nK; ++h) if (K[h]==47){ last=true; break; }
        if (last) { ++nLastLayer; continue; }
        if (primaryEnergy>=1.0 && primaryEnergy<=130.0) {
            gN1.push_back(N1); gN2.push_back(N2); gN3.push_back(N3);
            gEbeam.push_back(primaryEnergy);
        }
    }
    if (gN1.empty()) { std::cerr << "Pas de données exploitables.\n"; f->Close(); delete f; return; }

    // Paramètres initiaux dépendant de la particule
    double startParams[9];
    if (particle == "pi-") {
        startParams[0]=0.0430851;  startParams[1]=-3.50665e-05; startParams[2]=1.94847e-08;
        startParams[3]=0.107201;   startParams[4]=-6.36402e-05; startParams[5]=1.20235e-08;
        startParams[6]=1.81865e-13; startParams[7]=0.000735489; startParams[8]=-3.00925e-07;
    } else { // proton
        startParams[0]=0.0435267;  startParams[1]=-1.89745e-05; startParams[2]=1.07039e-08;
        startParams[3]=0.122286;   startParams[4]=-4.93952e-05; startParams[5]=5.45074e-09;
        startParams[6]=2.61942e-13; startParams[7]=0.00047743;  startParams[8]=-1.68624e-07;
    }

    // Minuit
    const char* parNames[9] = { "a0","a1","a2", "b0","b1","b2", "c0","c1","c2" };
    TMinuit* minuit = new TMinuit(9);
    minuit->SetFCN(chi2Function);
    minuit->DefineParameter(0, parNames[0], startParams[0], 1e-4, 0.0, 0.0);
    minuit->DefineParameter(1, parNames[1], startParams[1], 1e-6, 0.0, 0.0);
    minuit->DefineParameter(2, parNames[2], startParams[2], 1e-9, 0.0, 0.0);
    minuit->DefineParameter(3, parNames[3], startParams[3], 1e-4, 0.0, 0.0);
    minuit->DefineParameter(4, parNames[4], startParams[4], 1e-6, 0.0, 0.0);
    minuit->DefineParameter(5, parNames[5], startParams[5], 1e-6, 0.0, 0.0);
    minuit->DefineParameter(6, parNames[6], startParams[6], 1e-15, 0.0, 1e-10); // borne haute conservée
    minuit->DefineParameter(7, parNames[7], startParams[7], 1e-6, 0.0, 0.0);
    minuit->DefineParameter(8, parNames[8], startParams[8], 1e-6, 0.0, 0.0);
    minuit->Migrad();

    Double_t fmin=0, fedm=0, errdef=0; Int_t nvpar=0, nparx=0, istat=0;
    minuit->mnstat(fmin, fedm, errdef, nvpar, nparx, istat);
    const int nData = static_cast<int>(gN1.size());
    const int ndf   = nData - nvpar;
    std::cout << "=== Résultats minimisation ("<< particle <<") ===\n"
              << "chi2_min = " << fmin << "  ndf = " << ndf
              << "  chi2/ndf = " << (ndf>0 ? fmin/ndf : -1) << "\n"
              << "Evts filtrés (couche 47): " << nLastLayer << "\n"
              << "stat Minuit (istat) = " << istat << "  [0:no cov, 1:approx, 2:posdef, 3:accurate]\n";

    double par[9], err[9];
    for (int ip=0; ip<9; ++ip) minuit->GetParameter(ip, par[ip], err[ip]);

    // -------- Etrue/Ereco calculés une fois ----------
    std::vector<double> Etrue_all; Etrue_all.reserve(gN1.size());
    std::vector<double> Ereco_all; Ereco_all.reserve(gN1.size());
    for (size_t i = 0; i < gN1.size(); ++i) {
        const int    Nhit  = gN1[i] + gN2[i] + gN3[i];
        const double alpha = par[0] + par[1]*Nhit + par[2]*Nhit*Nhit;
        const double beta  = par[3] + par[4]*Nhit + par[5]*Nhit*Nhit;
        const double gamma = par[6] + par[7]*Nhit + par[8]*Nhit*Nhit;
        const double Ereco = alpha*gN1[i] + beta*gN2[i] + gamma*gN3[i];
        Etrue_all.push_back(gEbeam[i]);
        Ereco_all.push_back(Ereco);
    }

    // -------- Plots de base (plage fixe 1..130) ----------
    const std::string tag = "_" + particle;
    const double Emin_plot = 1.0;
    const double Emax_plot = 130.0;

    TH1D* hEreco = new TH1D(("hEreco"+tag).c_str(), "E_{reco};E_{reco} [GeV];N_{events}", 100, Emin_plot, Emax_plot);
    TH1D* hEbeam = new TH1D(("hEbeam"+tag).c_str(), "E_{beam};E_{beam} [GeV];N_{events}", 100, Emin_plot, Emax_plot);
    hEreco->SetDirectory(0); hEbeam->SetDirectory(0);

    TProfile* pProf = new TProfile(("pProf"+tag).c_str(),
                                   "Linearity profile;E_{true} [GeV];<E_{reco}> [GeV]",
                                   100, Emin_plot, Emax_plot);

    std::vector<double> x_true; x_true.reserve(gN1.size());
    std::vector<double> y_reco; y_reco.reserve(gN1.size());

    for (size_t i = 0; i < Etrue_all.size(); ++i) {
        hEreco->Fill(Ereco_all[i]);
        hEbeam->Fill(Etrue_all[i]);
        pProf->Fill(Etrue_all[i], Ereco_all[i]);
        x_true.push_back(Etrue_all[i]);
        y_reco.push_back(Ereco_all[i]);
    }

    TCanvas* cComp = new TCanvas(("cComp"+tag).c_str(), "Reco vs Beam", 800, 600);
    hEbeam->SetStats(kFALSE);
    hEbeam->SetLineColor(kBlue);
    hEreco->SetLineColor(kRed);
    hEbeam->Draw("HIST");
    hEreco->Draw("HIST SAME");
    {
        TLegend* leg = new TLegend(0.45,0.75,0.68,0.88);
        leg->AddEntry(hEbeam,"E Beam","l");
        leg->AddEntry(hEreco,"E reco","l");
        leg->Draw();
    }
    cComp->SaveAs(("plots/Reco_vs_Beam" + tag + ".png").c_str());

    TCanvas* c1 = new TCanvas(("c1"+tag).c_str(), "Linearity profile", 800, 600);
    pProf->SetStats(kFALSE);
    pProf->GetYaxis()->SetRangeUser(Emin_plot, Emax_plot);
    pProf->Draw();
    {
        TLine* l1 = new TLine(Emin_plot, Emin_plot, Emax_plot, Emax_plot);
        l1->SetLineColor(kRed); l1->SetLineStyle(2); l1->Draw();
    }
    c1->SaveAs(("plots/Profil_linearite" + tag + ".png").c_str());

    TCanvas* c2 = new TCanvas(("c2"+tag).c_str(), "Linearity", 800, 600);
    if (!x_true.empty()) {
        TGraph* g2 = new TGraph((int)x_true.size(), x_true.data(), y_reco.data());
        g2->SetTitle("E_{reco} vs E_{beam};E_{beam} [GeV];E_{reco} [GeV]");
        g2->GetXaxis()->SetLimits(Emin_plot, Emax_plot);
        g2->GetYaxis()->SetRangeUser(Emin_plot, Emax_plot);
        g2->Draw("AP");
        TLine* l2 = new TLine(Emin_plot, Emin_plot, Emax_plot, Emax_plot);
        l2->SetLineColor(kRed); l2->SetLineStyle(2); l2->SetLineWidth(3); l2->Draw();
    }
    c2->SaveAs(("plots/Linearity" + tag + ".png").c_str());

    // -------- Résolution relative (bins en Etrue sur [1,130]) ----------
    // Déterminer une fenêtre ΔE robuste via le 95e centile des résidus
    std::vector<double> residuals; residuals.reserve(Etrue_all.size());
    for (size_t i=0;i<Etrue_all.size();++i) residuals.push_back(Ereco_all[i]-Etrue_all[i]);
    double dE_range = 20.0;
    if (!residuals.empty()) {
        std::vector<double> tmp = residuals;
        size_t k = tmp.size()*95/100;
        std::nth_element(tmp.begin(), tmp.begin()+k, tmp.end());
        double p95 = std::fabs(tmp[k]);
        dE_range = std::max(20.0, 1.5*p95);
    }

    const int    nBins = 20;
    const double Emin  = Emin_plot;
    const double Emax  = Emax_plot;
    const double dEbin = (Emax - Emin)/nBins;

    std::vector<double> binCenter(nBins), sigma(nBins,0.0), sigmaErr(nBins,0.0);
    for (int ib = 0; ib < nBins; ++ib) {
        binCenter[ib] = Emin + (ib + 0.5)*dEbin;
        const double eLow  = Emin + ib*dEbin;
        const double eHigh = eLow + dEbin;

        TH1D hde(("hde"+tag+std::to_string(ib)).c_str(),
                 "ΔE;E_{reco}-E_{true} [GeV];N_{events}", 120, -dE_range, dE_range);
        int filled = 0;
        for (size_t i = 0; i < Etrue_all.size(); ++i) {
            if (Etrue_all[i] >= eLow && Etrue_all[i] < eHigh) {
                hde.Fill(Ereco_all[i] - Etrue_all[i]);
                ++filled;
            }
        }
        if (filled > 20) {
            hde.Fit("gaus","Q");
            TF1* f = hde.GetFunction("gaus");
            if (f) { sigma[ib] = f->GetParameter(2); sigmaErr[ib] = f->GetParError(2); }
        }
    }

    std::vector<double> relSigma(nBins,0.0), relErr(nBins,0.0);
    for (int ib = 0; ib < nBins; ++ib) {
        relSigma[ib] = (binCenter[ib]>0 ? sigma[ib] / binCenter[ib] : 0.0);
        relErr  [ib] = (binCenter[ib]>0 ? sigmaErr[ib] / binCenter[ib] : 0.0);
    }

    TGraphErrors* gRel = new TGraphErrors(nBins, binCenter.data(), relSigma.data(), 0, relErr.data());
    gRel->SetTitle("Relative resolution;E_{true} [GeV];#sigma/E");
    TCanvas* cRes = new TCanvas(("cRes"+tag).c_str(), "sigma/E vs E", 800, 600);
    gRel->GetXaxis()->SetLimits(Emin_plot, Emax_plot);
    gRel->Draw("APL");
    cRes->SaveAs(("plots/Resolution_relative" + tag + ".png").c_str());

    // Nettoyage
    delete minuit;
    f->Close();
    delete f;
}
