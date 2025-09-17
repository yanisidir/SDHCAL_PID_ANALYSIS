// compareRadius.C
#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TLegend.h>
#include <iostream>
#include <algorithm>

void compareRadius() {
    // 1) Ouvre les fichiers & récupère les arbres
    TFile* filePion   = TFile::Open("/gridgroup/ilc/midir/analyse/data/merged_primaryEnergy/200k_pi-_1-130_params_merged.root");
    TFile* fileProton = TFile::Open("/gridgroup/ilc/midir/analyse/data/merged_primaryEnergy/200k_proton_1-130_params_merged.root");
    // TFile* fileKaon   = TFile::Open("/home/ilc/midir/Timing/files/analyse/analyse_kaon/Params.root");

    if (!filePion || filePion->IsZombie() ||
        !fileProton || fileProton->IsZombie()
        // || !fileKaon || fileKaon->IsZombie()
       ) {
        std::cerr << "Erreur: impossible d'ouvrir un des fichiers" << std::endl;
        return;
    }

    TTree* tPion   = static_cast<TTree*>(filePion->Get("tree"));
    TTree* tProton = static_cast<TTree*>(fileProton->Get("tree"));
    // TTree* tKaon   = static_cast<TTree*>(fileKaon->Get("tree"));

    if (!tPion || !tProton) {
        std::cerr << "Erreur: arbre tree non trouvé" << std::endl;
        filePion->Close(); fileProton->Close();
        return;
    }

    // 2) Min/Max de Radius
    Double_t minP  = tPion->GetMinimum("meanRadius");
    Double_t maxP  = tPion->GetMaximum("meanRadius");
    Double_t minPr = tProton->GetMinimum("meanRadius");
    Double_t maxPr = tProton->GetMaximum("meanRadius");

    std::cout << "Pion Radius range:   [" << minP  << ", " << maxP  << "]" << std::endl;
    std::cout << "Proton Radius range: [" << minPr << ", " << maxPr << "]" << std::endl;

    // 3) Bornes & binning
    Double_t xmin = std::min(minP, minPr);
    // Double_t xmax = std::max(maxP, maxPr);
    Double_t xmax = 150;
    Int_t    nbins = 150 ; // marge pour flottants
    if (nbins < 1) {
        std::cerr << "Erreur: intervalle invalide (" << xmin << "," << xmax << ")" << std::endl;
        filePion->Close(); fileProton->Close();
        return;
    }
    std::cout << "Uniform binning: " << nbins << " bins sur [" << xmin << ", " << xmax << "]" << std::endl;

    // 4) Histos
    TH1D* hPion   = new TH1D("hPionRadius",   "Mean Radius;<Radius>;Evts", nbins, xmin, xmax);
    TH1D* hProton = new TH1D("hProtonRadius", ";Radius;Evts",                   nbins, xmin, xmax);

    // 5) Remplissage
    Double_t dRadius;

    tPion->SetBranchStatus("*", 0);       // tout désactiver
    tPion->SetBranchStatus("meanRadius", 1);    // activer Radius
    tPion->SetBranchAddress("meanRadius", &dRadius);
    for (Long64_t i = 0; i < tPion->GetEntries(); ++i) {
        tPion->GetEntry(i);
        hPion->Fill(dRadius);
    }

    tProton->SetBranchStatus("*", 0);
    tProton->SetBranchStatus("meanRadius", 1);
    tProton->SetBranchAddress("meanRadius", &dRadius);
    for (Long64_t i = 0; i < tProton->GetEntries(); ++i) {
        tProton->GetEntry(i);
        hProton->Fill(dRadius);
    }

    if (hPion->Integral() > 0)   hPion->Scale(1.0 / hPion->Integral());
    if (hProton->Integral() > 0) hProton->Scale(1.0 / hProton->Integral());

    // 6) Style & affichage
    hPion->SetLineColor(kRed);
    hPion->SetLineWidth(2);
    hProton->SetLineColor(kBlue);
    hProton->SetLineWidth(2);

    Double_t maxH = std::max({hPion->GetMaximum(), hProton->GetMaximum()}) * 1.2;
    hPion->SetMaximum(maxH);
    hProton->SetMaximum(maxH);

    TCanvas* c = new TCanvas("cRadius", "Comparaison Radius", 600, 600);
    c->SetGrid();
    hPion->SetStats(kFALSE);
    hPion->Draw("HIST");
    hProton->SetStats(kFALSE);
    hProton->Draw("HIST SAME");

    TLegend* leg = new TLegend(0.7, 0.75, 0.88, 0.88);
    leg->SetFillColorAlpha(0, 0.5);
    leg->AddEntry(hPion,  "Pions- ",   "L");
    leg->AddEntry(hProton,"Protons ", "L");
    leg->Draw();

    c->Update();
    c->SaveAs("Radius.pdf");

    // Nettoyage
    delete leg;
    delete hPion; delete hProton;
    delete c;
    filePion->Close(); fileProton->Close();
}
