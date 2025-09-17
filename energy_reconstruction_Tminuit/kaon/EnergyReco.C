// EnergyReco.C - Reconstruction de l\'energie primaire du faisceau

#include <vector>
#include <iostream>
#include <cmath>

#include "TMath.h"
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TGraph.h"
#include "TLine.h"
#include "TMinuit.h"

// Donnees globales pour la minimisation
std::vector<int>    gN1, gN2, gN3;
std::vector<double> gEdep, gEbeam;

// Fonction à minimiser : chi2 = somme_i (Ereco_i - Ebeam_i)^2 / Ebeam_i
void chi2Function(int& npar, double* grad, double& fval, double* par, int flag) {
    const double bigPenalty = 1e30;
    double chi2 = 0.0;
    for (size_t i = 0; i < gN1.size(); ++i) {
        if (gEbeam[i] <= 0.0) continue;  // on elimine les Ebeam <= 0

        int Nhit = gN1[i] + gN2[i] + gN3[i];
        double alpha = par[0] + par[1]*Nhit + par[2]*Nhit*Nhit;
        double beta  = par[3] + par[4]*Nhit + par[5]*Nhit*Nhit;
        double gamma = par[6] + par[7]*Nhit + par[8]*Nhit*Nhit;

        double Ereco = alpha*gN1[i] + beta*gN2[i] + gamma*gN3[i];
        double diff = Ereco - gEbeam[i];
        chi2 += diff*diff / gEbeam[i];
    }
    if (!std::isfinite(chi2)) chi2 = bigPenalty;
    fval = chi2;
}

void EnergyReco() {
    // -----------------------------------------------------------------------
    // 1) Ouverture des fichiers
    // -----------------------------------------------------------------------

    TFile* fHits   = TFile::Open("/gridgroup/ilc/midir/analyse/data/digitized/130k_kaon_E1to130_digitized.root");
    TFile* fEnergy = TFile::Open("/gridgroup/ilc/midir/analyse/data/raw/130k_kaon_E1to130.root");
    if (!fHits || !fEnergy) {
        std::cerr << "Erreur à l'ouverture des fichiers ROOT !" << std::endl;
        return;
    }

    // -----------------------------------------------------------------------
    // 2) Recuperation des TTrees
    // -----------------------------------------------------------------------
    TTree* treeHits   = (TTree*)fHits->Get("tree");
    TTree* treeEnergy = (TTree*)fEnergy->Get("tree");

    // --- Branches hits ---
    std::vector<int>* vThr = nullptr;
    std::vector<int>* vK   = nullptr;
    treeHits->SetBranchAddress("thr", &vThr);
    treeHits->SetBranchAddress("K",   &vK);

    // --- Branche energie deposee et primaire ---
    double depenergy = 0;
    double primenergy = 0;
    treeEnergy->SetBranchAddress("depositedEnergy", &depenergy);
    treeEnergy->SetBranchAddress("primaryEnergy",   &primenergy);

    // -----------------------------------------------------------------------
    // 3) Boucle sur les evenements – preparation des tableaux globaux
    // -----------------------------------------------------------------------

    const int kLastLayer = 47;
    int nLastLayer = 0;

    // --- Boucle sur les evenements ---
    Long64_t nEntries = treeHits->GetEntries();
    for (Long64_t i = 0; i < nEntries; ++i) {
        treeHits->GetEntry(i);
        treeEnergy->GetEntry(i);

        // Filtre : hit dans la dernière couche
        bool hasLastLayerHit = false;
        for (int k : *vK) {
            if (k == kLastLayer) { hasLastLayerHit = true; break; }
        }
        if (hasLastLayerHit) { ++nLastLayer; continue; }

        // Comptage par seuil
        int N1 = 0, N2 = 0, N3 = 0;
        for (size_t h = 0; h < vThr->size(); ++h) {
            switch ((*vThr)[h]) {
                case 1: ++N1; break;
                case 2: ++N2; break;
                case 3: ++N3; break;
                default: break;
            }
        }

        // Stocke si energie primaire > 0
        if (primenergy > 0.0) {
            gN1.push_back(N1);
            gN2.push_back(N2);
            gN3.push_back(N3);
            gEdep.push_back(depenergy);
            gEbeam.push_back(primenergy);
        }
    }

    // -----------------------------------------------------------------------
    // 4) Minuit : definition des paramètres et fit
    // -----------------------------------------------------------------------
    TMinuit* minuit = new TMinuit(9);
    minuit->SetFCN(chi2Function);
    const char* parNames[9] = { "a0","a1","a2", "b0","b1","b2", "c0","c1","c2" };
    
    // --- paramètres initiaux et steps ---
    // Paramètre a0
    minuit->DefineParameter(0, parNames[0], /*start=*/0.0560298, /*step=*/1e-5, /*min=*/0.0, /*max=*/0.0);
    // Paramètre a1
    minuit->DefineParameter(1, parNames[1], /*start=*/-4.58223e-05, /*step=*/1e-5,  /*min=*/0.0, /*max=*/0.0);
    // Paramètre a2
    minuit->DefineParameter(2, parNames[2], /*start=*/1.90941e-08, /*step=*/1e-5,  /*min=*/0.0, /*max=*/0.0);

    // Paramètre b0
    minuit->DefineParameter(3, parNames[3],  /*start=*/0.0800358, /*step=*/1e-5,  /*min=*/0.0, /*max=*/0.0);  
    // Paramètre b1
    minuit->DefineParameter(4, parNames[4],  /*start=*/-7.76142e-05, /*step=*/1e-5,  /*min=*/0.0, /*max=*/0.0);
    // Paramètre b2
    minuit->DefineParameter(5, parNames[5],  /*start=*/4.5824e-08, /*step=*/1e-5,  /*min=*/0.0, /*max=*/0.0);

    // Paramètre c0
    minuit->DefineParameter(6, parNames[6],  /*start=*/1.78598e-14, /*step=*/1e-5,   /*min=*/1e-10, /*max=*/0.0);  
    // Paramètre c1
    minuit->DefineParameter(7, parNames[7],  /*start=*/0.000797643, /*step=*/1e-5,  /*min=*/0.0, /*max=*/0.0);
    // Paramètre c2
    minuit->DefineParameter(8, parNames[8], /*start=*/-3.5693e-07, /*step=*/1e-5, /*min=*/0.0, /*max=*/0.0);  // 
    minuit->Migrad();

    Double_t fmin, fedm, errdef; Int_t nvpar, nparx, istat;
    minuit->mnstat(fmin, fedm, errdef, nvpar, nparx, istat);
    if (istat != 3) std::cerr << "Statut Minuit = "<<istat<<" (3 = bon)\n";

    int nData = gN1.size();
    int ndf   = nData - nvpar;
    std::cout << "=== Resultats minimisation ===\n"
              << "χ²_min = "<<fmin<<"  ndf="<<ndf<<"  χ²/ndf="<<(fmin/ndf)<<"\n"
              << "Evts filtres (couche 47): "<<nLastLayer<<"\n";

    double par[9], err[9];
    for (int ip = 0; ip < 9; ++ip) minuit->GetParameter(ip, par[ip], err[ip]);
    std::cout << "Paramètres optimises :\n";
    for (int ip = 0; ip < 9; ++ip) std::cout << parNames[ip] << " = " << par[ip] << std::endl;

    // -----------------------------------------------------------------------
    // 5) Histogrammes / graphes : linearite
    // -----------------------------------------------------------------------
    TH1D* hEreco = new TH1D("hEreco","E_{reco} ;E_{reco} [GeV];N_{events}",100,0,100);
    TH1D* hEbeam = new TH1D("hEbeam","E beam vs E reco;E [GeV];N_{events}",100,0,100);
    TProfile* pProf = new TProfile("pProf","Linearity profile ;E_{beam true} [GeV];<E_{reco}> [GeV]",100,0,100);

    std::vector<double> x_true, y_reco;
    for (size_t i = 0; i < gN1.size(); ++i) {
        // on ne trace que jusqu'à 100 GeV
        if (gEbeam[i] > 100.0) continue;

        int Nhit = gN1[i] + gN2[i] + gN3[i];
        double alpha = par[0] + par[1]*Nhit + par[2]*Nhit*Nhit;
        double beta  = par[3] + par[4]*Nhit + par[5]*Nhit*Nhit;
        double gamma = par[6] + par[7]*Nhit + par[8]*Nhit*Nhit;
        double Ereco = alpha*gN1[i] + beta*gN2[i] + gamma*gN3[i];
        hEreco->Fill(Ereco);
        hEbeam->Fill(gEbeam[i]);
        pProf->Fill(gEbeam[i], Ereco);
        x_true.push_back(gEbeam[i]);
        y_reco.push_back(Ereco);
    }

    TCanvas* cComp = new TCanvas("cComp","Reco vs Beam",800,600);
    hEbeam->SetStats(kFALSE);
    hEbeam->SetLineColor(kBlue);
    hEreco->SetLineColor(kRed);
    hEbeam->Draw("HIST");
    hEreco->Draw("HIST SAME");
    auto leg = new TLegend(0.45,0.75,0.68,0.88);
    leg->AddEntry(hEbeam,"E Beam","l");
    leg->AddEntry(hEreco,"E reco","l");
    leg->Draw();
    cComp->SaveAs("plots/Reco_vs_Beam_kaon.pdf");

    TCanvas* c1 = new TCanvas("c1","Linearity profile",800,600);
    pProf->SetStats(kFALSE);
    pProf->GetYaxis()->SetRangeUser(0, 100);
    pProf->Draw();
    TLine* l1 = new TLine(0,0,100,100); l1->SetLineColor(kRed); l1->SetLineStyle(2); l1->Draw();
    c1->SaveAs("plots/Profil_linearite_kaon.pdf");

    TCanvas* c2 = new TCanvas("c2","Linearity",800,600);
    TGraph* g2 = new TGraph(x_true.size(), &x_true[0], &y_reco[0]);
    g2->SetTitle("E reco vs E Beam ;E_{beam} [GeV];E_{reco} [GeV]");
    g2->GetYaxis()->SetRangeUser(0, 100);
    g2->Draw("AP");
    TLine* l2 = new TLine(0,0,100,100); l2->SetLineColor(kRed); l2->SetLineStyle(2); l2->SetLineWidth(3); l2->Draw();
    c2->SaveAs("plots/Linearity_kaon.pdf");

    // -----------------------------------------------------------------------
    // 6) Resolution sigma/E
    // -----------------------------------------------------------------------
    std::vector<double> Etrue_all, Ereco_all;
    for (size_t i = 0; i < gN1.size(); ++i) {
        const int    Nhit  = gN1[i] + gN2[i] + gN3[i];
        const double alpha = par[0] + par[1]*Nhit + par[2]*Nhit*Nhit;
        const double beta  = par[3] + par[4]*Nhit + par[5]*Nhit*Nhit;
        const double gamma = par[6] + par[7]*Nhit + par[8]*Nhit*Nhit;
        const double Ereco = alpha*gN1[i] + beta*gN2[i] + gamma*gN3[i];

        Etrue_all.push_back(gEbeam[i]);
        Ereco_all.push_back(Ereco);
    }

    constexpr int    nBins = 20;
    constexpr double Emin  =  0.;
    constexpr double Emax  = 100.;
    const double     dE    = (Emax - Emin)/nBins;

    std::vector<double> binCenter(nBins), sigma(nBins), sigmaErr(nBins);
    for (int ib = 0; ib < nBins; ++ib) {
        binCenter[ib] = Emin + (ib + 0.5)*dE;

        const double eLow  = Emin + ib*dE;
        const double eHigh = eLow + dE;

        TH1D hde("hde","ΔE;E_{reco}-E_{true} [GeV];N_{events}",100,-20,20);
        for (size_t i = 0; i < Etrue_all.size(); ++i)
            if (Etrue_all[i] >= eLow && Etrue_all[i] < eHigh)
                hde.Fill(Ereco_all[i] - Etrue_all[i]);

        hde.Fit("gaus","Q");
        TF1* f = hde.GetFunction("gaus");
        sigma   [ib] = f->GetParameter(2);
        sigmaErr[ib] = f->GetParError(2);
    }

    std::vector<double> relSigma(nBins), relErr(nBins);
    for (int ib = 0; ib < nBins; ++ib) {
        relSigma[ib] = sigma   [ib] / binCenter[ib];
        relErr  [ib] = sigmaErr[ib] / binCenter[ib];
    }

    TGraphErrors* gRel = new TGraphErrors(nBins,
                                          binCenter.data(), relSigma.data(),
                                          nullptr,          relErr.data());
    gRel->SetTitle("Relative resolution;E_{true} [GeV];sigma/E");

    TCanvas* cRes = new TCanvas("cRes","sigma/E vs E",800,600);
    gRel->Draw("APL");
    cRes->SaveAs("plots/Resolution_relative_kaon.pdf");
}

