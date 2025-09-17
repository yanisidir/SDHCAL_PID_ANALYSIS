// compareemFraction.C
#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TLegend.h>
#include <iostream>
#include <algorithm>

void compareEmFraction() {
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

    // 2) Min/Max de emFraction
    Double_t minP  = tPion->GetMinimum("emFraction");
    Double_t maxP  = tPion->GetMaximum("emFraction");
    Double_t minPr = tProton->GetMinimum("emFraction");
    Double_t maxPr = tProton->GetMaximum("emFraction");

    std::cout << "Pion emFraction range:   [" << minP  << ", " << maxP  << "]" << std::endl;
    std::cout << "Proton emFraction range: [" << minPr << ", " << maxPr << "]" << std::endl;

    // 3) Bornes & binning
    Double_t xmin = std::min(minP, minPr);
    // Double_t xmax = std::max(maxP, maxPr);
    Double_t xmax = 1.0001;
    Int_t    nbins = static_cast<Int_t>(xmax - xmin) + 50; // marge pour flottants
    if (nbins < 1) {
        std::cerr << "Erreur: intervalle invalide (" << xmin << "," << xmax << ")" << std::endl;
        filePion->Close(); fileProton->Close();
        return;
    }
    std::cout << "Uniform binning: " << nbins << " bins sur [" << xmin << ", " << xmax << "]" << std::endl;

    // 4) Histos
    TH1D* hPion   = new TH1D("hPionemFraction",   "emFraction;emFraction;Evts", nbins, xmin, xmax);
    TH1D* hProton = new TH1D("hProtonemFraction", ";emFraction;Evts",                   nbins, xmin, xmax);

    // 5) Remplissage
    double_t demFraction;

    tPion->SetBranchStatus("*", 0);       // tout désactiver
    tPion->SetBranchStatus("emFraction", 1);    // activer emFraction
    tPion->SetBranchAddress("emFraction", &demFraction);
    for (Long64_t i = 0; i < tPion->GetEntries(); ++i) {
        tPion->GetEntry(i);
        hPion->Fill(demFraction);
    }

    tProton->SetBranchStatus("*", 0);
    tProton->SetBranchStatus("emFraction", 1);
    tProton->SetBranchAddress("emFraction", &demFraction);
    for (Long64_t i = 0; i < tProton->GetEntries(); ++i) {
        tProton->GetEntry(i);
        hProton->Fill(demFraction);
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

    TCanvas* c = new TCanvas("cemFraction", "Comparaison emFraction", 600, 600);
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
    c->SaveAs("emFraction.pdf");

    // Nettoyage
    delete leg;
    delete hPion; delete hProton;
    delete c;
    filePion->Close(); fileProton->Close();
}
