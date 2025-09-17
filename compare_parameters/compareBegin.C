// compareBegin_refactored_three.C
#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TLegend.h>
#include <iostream>
#include <algorithm>


void compareBegin() {
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
        // fileKaon->Close();
        return;
    }

    // 2) Récupère min/max directement via ROOT
    Double_t minP  = tPion->GetMinimum("Begin");
    Double_t maxP  = tPion->GetMaximum("Begin");

    Double_t minPr = tProton->GetMinimum("Begin");
    Double_t maxPr = tProton->GetMaximum("Begin");

    // Double_t minK  = tKaon->GetMinimum("Begin");
    // Double_t maxK  = tKaon->GetMaximum("Begin");

    std::cout << "Pion Begin range:   [" << minP  << ", " << maxP  << "]" << std::endl;
    std::cout << "Proton Begin range: [" << minPr << ", " << maxPr << "]" << std::endl;
    // std::cout << "Kaon Begin range:   [" << minK  << ", " << maxK  << "]" << std::endl;

    // 3) Détermine bornes communes et binning uniforme
    Double_t xmin = 0;   // std::min(std::min(minP,  minPr), minK);
    Double_t xmax = 25;  // std::max(std::max(maxP,  maxPr), maxK);
    Int_t    nbins = static_cast<Int_t>(xmax - xmin) + 2;
    if (nbins < 1) {
        std::cerr << "Erreur: intervalle invalide (" << xmin << "," << xmax << ")" << std::endl;
        filePion->Close(); fileProton->Close(); 
        // fileKaon->Close();
        return;
    }
    std::cout << "Uniform binning: " << nbins << " bins sur [" 
              << xmin << ", " << xmax+1 << "]" << std::endl;

    // 4) Création des histogrammes
    TH1D* hPion   = new TH1D("hPion",   " Begin;Begin;Evts", nbins, xmin, xmax+1);
    TH1D* hProton = new TH1D("hProton", ";Begin;Evts",               nbins, xmin, xmax+1);
    // TH1D* hKaon   = new TH1D("hKaon",   ";Begin;Evts",               nbins, xmin, xmax+1);

    // 5) Remplissage
    Int_t dBegin;

    tPion->SetBranchStatus("*", 0);      
    tPion->SetBranchStatus("Begin", 1);    
    tPion->SetBranchAddress("Begin", &dBegin);
    for (Long64_t i = 0; i < tPion->GetEntries(); ++i) {
        tPion->GetEntry(i);
        hPion->Fill(dBegin);
    }

    tProton->SetBranchStatus("*", 0);
    tProton->SetBranchStatus("Begin", 1);    
    tProton->SetBranchAddress("Begin", &dBegin);
    for (Long64_t i = 0; i < tProton->GetEntries(); ++i) {
        tProton->GetEntry(i);
        hProton->Fill(dBegin);
    }
    // tKaon->SetBranchAddress("Begin", &dBegin);
    // for (Long64_t i = 0; i < tKaon->GetEntries(); ++i) {
    //     tKaon->GetEntry(i);
    //     hKaon->Fill(dBegin);
    // }

    if (hPion->Integral() > 0)   hPion->Scale(1.0 / hPion->Integral());
    if (hProton->Integral() > 0) hProton->Scale(1.0 / hProton->Integral());

    // 6) Style & affichage
    hPion->SetLineColor(kRed);
    hPion->SetLineWidth(2);
    hProton->SetLineColor(kBlue);
    hProton->SetLineWidth(2);
    // hKaon->SetLineColor(kGreen+2);
    // hKaon->SetLineWidth(2);
    Double_t maxH = std::max({hPion->GetMaximum(), hProton->GetMaximum()}) * 1.2;
    // Double_t maxH = std::max({hPion->GetMaximum(), hProton->GetMaximum(), hKaon->GetMaximum()}) * 1.2;
    hPion->SetMaximum(maxH);
    hProton->SetMaximum(maxH);
    // hKaon->SetMaximum(maxH);

    TCanvas* c = new TCanvas("c", "Comparaison Begin", 600, 600);
    // c->SetLogy();
    c->SetGrid();
    hPion->SetStats(kFALSE);
    hPion->Draw("HIST");
    hProton->SetStats(kFALSE);
    hProton->Draw("HIST SAME");
    // hKaon->SetStats(kFALSE);
    // hKaon->Draw("HIST SAME");

    // // 7) Fit exponentiel
    // auto makeExpoFit = [&](TH1D* h, const char* name, Color_t col) {
    //     Double_t m = h->GetMean(), r = h->GetRMS();
    //     TF1* f = new TF1(name, "[0]*exp(-x/[1])", m - 2*r, m + 2*r);
    //     f->SetParameters(h->GetMaximum(), r);
    //     h->Fit(f, "R+");
    //     f->SetLineColor(col);
    //     f->SetLineWidth(2);
    //     f->SetLineStyle(2);
    //     f->Draw("SAME");
    //     return f;
    // };
    // TF1* fitPion = makeExpoFit(hPion,   "fitExpPion",   kRed);
    // TF1* fitProt = makeExpoFit(hProton, "fitExpProt",   kBlue);
    // TF1* fitKaon = makeExpoFit(hKaon,   "fitExpKaon",   kGreen+2);

    // 8) Légende et sauvegarde
    TLegend* leg = new TLegend(0.7, 0.75, 0.88, 0.88);
    leg->SetFillColorAlpha(0, 0.5);
    leg->AddEntry(hPion,  "Pions- ",   "L");
    leg->AddEntry(hProton,"Protons ", "L");
    // leg->AddEntry(hKaon,  "Kaons ",   "L");
    // leg->AddEntry(fitPion,"Fit exp pions",  "L");
    // leg->AddEntry(fitProt,"Fit exp protons","L");
    // leg->AddEntry(fitKaon,"Fit exp kaons",  "L");
    leg->Draw();

    c->Update();
    c->SaveAs("Begin.pdf");

    // 9) Nettoyage mémoire
    delete leg;
    // delete fitPion; delete fitProt; delete fitKaon;
    delete hPion; delete hProton; 
    // delete hKaon;
    delete c;
    filePion->Close(); fileProton->Close(); 
    // fileKaon->Close();
}
