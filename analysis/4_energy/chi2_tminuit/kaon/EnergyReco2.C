// EnergyReco2.C

#include <vector>
#include <iostream>
#include <cmath>

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TGraph.h"
#include "TLine.h"
#include "TMinuit.h"



// Donnees globales pour la minimisation
std::vector<int> gN1, gN2, gN3;
std::vector<double> gEdep;


void EnergyReco2() {
    // --- Ouverture des fichiers ROOT ---
    TFile* fHits   = TFile::Open("data/100k_kaon_merged_digitized.root");
    TFile* fEnergy = TFile::Open("data/100k_kaon_merged.root");
    if (!fHits || !fEnergy) {
        std::cerr << "Erreur à l'ouverture des fichiers ROOT !" << std::endl;
        return;
    }

    // --- Recuperation des TTrees ---
    TTree* treeHits   = (TTree*)fHits->Get("tree");
    TTree* treeEnergy = (TTree*)fEnergy->Get("tree");

    // --- Branches hits ---
    std::vector<int>*    vThr = nullptr;

    treeHits->SetBranchAddress("thr", &vThr);

    // --- Branche energie deposee ---
    double energy = 0;
    treeEnergy->SetBranchAddress("depositedEnergy", &energy);

    Long64_t nEvt = treeHits->GetEntries();
    gN1.reserve(nEvt);
    gN2.reserve(nEvt);
    gN3.reserve(nEvt);
    gEdep.reserve(nEvt);

    // --- Boucle sur les evenements ---
    for (Long64_t i = 0; i < nEvt; ++i) {
        treeHits->GetEntry(i);
        treeEnergy->GetEntry(i);

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

        gN1.push_back(N1);
        gN2.push_back(N2);
        gN3.push_back(N3);
        gEdep.push_back(energy);
    }

    double par[9] = {
    0.0439714, -5.6607e-05, 3.24703e-08, 0.087253, -9.48706e-05, 5.67426e-08, 1.65017e-14, 0.000919808, -5.26005e-07};   

    double Par2[9] = {
    0.0623032, -6.94959e-05, 2.83338e-08, 0.0625425, -6.27755e-05, 1.01988e-07, 9.98157e-11, 0.00117327, -7.59878e-07};

    // --- Creation des histogrammes comparatifs ---
    TH1D* hEdep = new TH1D("hEdep","Comparaison Energies reconstruites, deposee;E [GeV];N_{events}",  100, 0, 100);
    TH1D* hEreco = new TH1D("hEreco", "Energie reconstruite;E_{reco} [GeV];N_{events}", 100, 0, 100);
    TH1D* hEreco2 = new TH1D("hEreco2", "Energie reconstruite comparative;E_{reco} [GeV];N_{events}", 100, 0, 100);

    // Vecteurs pour la linearite
    std::vector<double> x_dep, y_reco, y_2;
    x_dep.reserve(gEdep.size());
    y_reco .reserve(gEdep.size());
    y_2.reserve(gEdep.size());

    // --- Remplissage des deux ensembles ---
    for (size_t i = 0; i < gN1.size(); ++i) {
        int Nhit = gN1[i] + gN2[i] + gN3[i];

        // reco 
        double alpha = par[0] + par[1]*Nhit + par[2]*Nhit*Nhit;
        double beta  = par[3] + par[4]*Nhit + par[5]*Nhit*Nhit;
        double gamma = par[6] + par[7]*Nhit + par[8]*Nhit*Nhit;
        double E_reco = alpha*gN1[i] + beta*gN2[i] + gamma*gN3[i];
        hEreco  ->Fill(E_reco);
        y_reco .push_back(E_reco);

        // reco comparatif
        double aF = Par2[0] + Par2[1]*Nhit + Par2[2]*Nhit*Nhit;
        double bF = Par2[3] + Par2[4]*Nhit + Par2[5]*Nhit*Nhit;
        double cF = Par2[6] + Par2[7]*Nhit + Par2[8]*Nhit*Nhit;
        double E_reco2  = aF*gN1[i] + bF*gN2[i] + cF*gN3[i];
        hEreco2->Fill(E_reco2);
        y_2.emplace_back(E_reco2);

        x_dep.emplace_back(gEdep[i]);

        hEdep ->Fill(gEdep[i]);
    }

    // --- Style des histogrammes ---
    hEdep->SetLineColor(kBlack);
    hEdep->SetLineWidth(2);

    hEreco->SetLineColor(kBlue);
    hEreco->SetLineWidth(2);

    hEreco2->SetLineColor(kRed);
    hEreco2->SetLineWidth(2);    

    // --- Dessin des histogrammes superposes ---
    TCanvas* c1 = new TCanvas("c1","Comparaison des reco",800,600);
    hEdep->SetStats(kFALSE);
    hEdep->Draw();
    hEreco->Draw("SAME");
    hEreco2->Draw("SAME");
    auto leg = new TLegend(0.35,0.75,0.58,0.88);
    leg->AddEntry(hEdep,  "energie deposee", "l");
    leg->AddEntry(hEreco, "energie reco (par defaut)", "l");
    leg->AddEntry(hEreco2,"energie reco comparative",   "l");
    leg->SetBorderSize(0);
    leg->Draw();    

    // --- Graphe linearite comparatif ---
    TCanvas* c2 = new TCanvas("c2","Linearite comparee",800,600);
    TMultiGraph* mg = new TMultiGraph();
    TGraph* g   = new TGraph(x_dep.size(), &x_dep[0], &y_reco[0]);
    TGraph* g2 = new TGraph(x_dep.size(), &x_dep[0], &y_2[0]);
    // g  ->SetMarkerStyle(20);
    // g2->SetMarkerStyle(21);
    g2->SetMarkerColor(kBlue);
    mg->Add(g,"P");
    mg->Add(g2,"P");
    mg->SetTitle("Linearite comparee;E_{dep} [GeV];E_{reco} [GeV]");
    mg->Draw("A");
    TLine* line = new TLine(0,0,100,100);
    line->SetLineStyle(1);
    line->SetLineColor(kRed);
    line->Draw();
    auto leg2 = new TLegend(0.25,0.75,0.48,0.88);
    leg2->AddEntry(g,  "Energie reco", "lp");
    leg2->AddEntry(g2, "Energie reco comparative", "lp");
    leg2->SetBorderSize(0);
    leg2->Draw();
}

