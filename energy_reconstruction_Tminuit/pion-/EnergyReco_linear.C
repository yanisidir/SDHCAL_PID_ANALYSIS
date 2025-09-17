// EnergyReco_linear.C - Reconstruction lineaire de l'energie primaire

#include <vector>
#include <iostream>
#include <cmath>

#include "TMath.h"
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TProfile.h"
#include "TGraph.h"
#include "TLine.h"
#include "TMinuit.h"

// Donnees globales pour la minimisation
std::vector<int>    gN1, gN2, gN3;
std::vector<double> gEdep, gEbeam;

// Nouvelle fonction à minimiser : paramètres constants alpha, beta, gamma
void chi2Function(int& npar, double* grad, double& fval, double* par, int flag) {
    const double bigPenalty = 1e30;
    double chi2 = 0.0;
    // par[0] = alpha, par[1] = beta, par[2] = gamma
    double alpha = par[0];
    double beta  = par[1];
    double gamma = par[2];

    for (size_t i = 0; i < gN1.size(); ++i) {
        double Ebeam = gEbeam[i];
        if (Ebeam <= 0.0) continue;

        double Ereco = alpha * gN1[i]
                     + beta  * gN2[i]
                     + gamma * gN3[i];

        double diff = Ereco - Ebeam;
        chi2 += (diff * diff) / Ebeam;
    }
    if (!std::isfinite(chi2)) chi2 = bigPenalty;
    fval = chi2;
}

void EnergyReco_linear() {
    // 1) Ouverture des fichiers
    TFile* fHits   = TFile::Open("data/130k_pi_E1to130_digitized.root");
    TFile* fEnergy = TFile::Open("data/130k_pi_E1to130.root");
    if (!fHits || !fEnergy) {
        std::cerr << "Erreur à l'ouverture des fichiers ROOT !" << std::endl;
        return;
    }

    // 2) Recuperation des TTrees
    TTree* treeHits   = (TTree*)fHits->Get("tree");
    TTree* treeEnergy = (TTree*)fEnergy->Get("tree");

    std::vector<int>* vThr = nullptr;
    std::vector<int>* vK   = nullptr;
    treeHits->SetBranchAddress("thr", &vThr);
    treeHits->SetBranchAddress("K",   &vK);

    double depenergy = 0;
    double primenergy = 0;
    treeEnergy->SetBranchAddress("depositedEnergy", &depenergy);
    treeEnergy->SetBranchAddress("primaryEnergy",   &primenergy);

    // 3) Boucle sur les evenements
    const int kLastLayer = 47;
    int nLastLayer = 0;
    Long64_t nEntries = treeHits->GetEntries();
    for (Long64_t i = 0; i < nEntries; ++i) {
        treeHits->GetEntry(i);
        treeEnergy->GetEntry(i);

        bool hasLastLayerHit = false;
        for (int k : *vK) {
            if (k == kLastLayer) { hasLastLayerHit = true; break; }
        }
        if (hasLastLayerHit) { ++nLastLayer; continue; }

        int N1 = 0, N2 = 0, N3 = 0;
        for (size_t h = 0; h < vThr->size(); ++h) {
            switch ((*vThr)[h]) {
                case 1: ++N1; break;
                case 2: ++N2; break;
                case 3: ++N3; break;
                default: break;
            }
        }

        if (primenergy > 0.0) {
            gN1.push_back(N1);
            gN2.push_back(N2);
            gN3.push_back(N3);
            gEdep.push_back(depenergy);
            gEbeam.push_back(primenergy);
        }
    }

    // 4) Minimisation TMinuit avec 3 paramètres
    TMinuit* minuit = new TMinuit(3);
    minuit->SetFCN(chi2Function);
    const char* parNames[3] = { "alpha", "beta", "gamma" };

    minuit->DefineParameter(0, parNames[0], 0.1, 1e-4, 0.0, 0.0);
    minuit->DefineParameter(1, parNames[1], 0.1, 1e-4, 0.0, 0.0);
    minuit->DefineParameter(2, parNames[2], 0.1, 1e-4, 0.0, 0.0);
    minuit->Migrad();

    Double_t fmin; Int_t nvpar, nparx, istat;
    Double_t fedm, errdef;
    minuit->mnstat(fmin, fedm, errdef, nvpar, nparx, istat);
    if (istat != 3) std::cerr << "Statut Minuit = "<<istat<<" (3 = bon)\n";

    int nData = gN1.size();
    int ndf   = nData - nvpar;
    std::cout << "=== Resultats minimisation ===\n"
              << "χ²_min = "<<fmin<<"  ndf="<<ndf<<"  χ²/ndf="<<(fmin/ndf)<<"\n"
              << "Evts filtres (couche 47): "<<nLastLayer<<"\n";

    double par[3], err[3];
    for (int ip = 0; ip < 3; ++ip) minuit->GetParameter(ip, par[ip], err[ip]);
    std::cout << "Paramètres optimises :\n";
    for (int ip = 0; ip < 3; ++ip)
        std::cout << parNames[ip] << " = " << par[ip] << " ± " << err[ip] << std::endl;

    // 5) Évaluation linearite et resolution (similaire à la version precedente)
    TH1D* hEreco = new TH1D("hEreco","E_{reco} ;E_{reco} [GeV];N_{events}",100,0,100);
    TH1D* hEbeam = new TH1D("hEbeam","E beam vs E reco;E [GeV];N_{events}",100,0,100);
    TProfile* pProf = new TProfile("pProf","Profil linearite;E_{beam true} [GeV];<E_{reco}> [GeV]",100,0,100);
    std::vector<double> x_true, y_reco;

    for (size_t i = 0; i < gN1.size(); ++i) {
        if (gEbeam[i] > 100.0) continue;
        double Ereco = par[0]*gN1[i] + par[1]*gN2[i] + par[2]*gN3[i];
        hEreco->Fill(Ereco);
        hEbeam->Fill(gEbeam[i]);
        pProf->Fill(gEbeam[i], Ereco);
        x_true.push_back(gEbeam[i]);
        y_reco.push_back(Ereco);
    }

    TCanvas c1("c1","Linearite",800,600);
    hEbeam->SetStats(kFALSE);
    hEbeam->SetLineColor(kBlue);
    hEreco->SetLineColor(kRed);
    hEbeam->Draw("HIST");
    hEreco->Draw("HIST SAME");
    TLegend leg(0.45,0.75,0.68,0.88);
    leg.AddEntry(hEbeam,"E Beam","l");
    leg.AddEntry(hEreco,"E reco","l");
    leg.Draw();
    c1.SaveAs("plots/Reco_vs_Beam_lineaire_pi.pdf");

    TCanvas c2("c2","Profil linearite",800,600);
    pProf->SetStats(kFALSE);
    pProf->GetYaxis()->SetRangeUser(0, 100);
    pProf->Draw();
    TLine l(0,0,100,100); l.SetLineColor(kRed); l.SetLineStyle(2); l.Draw();
    c2.SaveAs("plots/Profil_linearite_lineaire_pi.pdf");

    // Resolution relative
    const int nBins = 20;
    const double Emin = 0., Emax = 100.;
    const double dE = (Emax - Emin)/nBins;
    std::vector<double> binCenter(nBins), relSigma(nBins), relErr(nBins);

    for (int ib = 0; ib < nBins; ++ib) {
        double eLow = Emin + ib*dE;
        double eHigh = eLow + dE;
        TH1D hres("hres","Residus;E_{reco}-E_{true};N",100,-20,20);
        for (size_t i = 0; i < x_true.size(); ++i) {
            if (x_true[i] >= eLow && x_true[i] < eHigh) {
                double diff = y_reco[i] - x_true[i];
                hres.Fill(diff);
            }
        }
        hres.Fit("gaus","Q");
        TF1* f = hres.GetFunction("gaus");
        double sigma = f->GetParameter(2);
        double errsigma = f->GetParError(2);
        binCenter[ib] = Emin + (ib + 0.5)*dE;
        relSigma[ib] = sigma / binCenter[ib];
        relErr[ib]   = errsigma / binCenter[ib];
    }
    TGraphErrors gRes(nBins, binCenter.data(), relSigma.data(), nullptr, relErr.data());
    TCanvas c3("c3","Resolution relative",800,600);
    gRes.SetTitle("Resolution relative;E_{true} [GeV];σ/E");
    gRes.Draw("APL");
    c3.SaveAs("plots/Resolution_relative_lineaire_pi.pdf");
}
