// addParam.C 
// ouvre Params.root et 100k_kaon_merged_digitized.root,


#include <iostream>
#include "TFile.h"
#include "TTree.h"

void addParam() {
    std::cout << "[debug] Début de addParam()\n";

    // 1) Ouvrir Params.root
    std::cout << "[debug] 1) Ouverture de Params.root...\n";
    TFile* f1 = TFile::Open("data/Params.root", "UPDATE");
    if (!f1 || f1->IsZombie()) {
        std::cerr << "## ERREUR [step1] : impossible d'ouvrir Params.root\n";
        return;
    }
    std::cout << "[debug]    Params.root ouvert avec succès.\n";

    // Récupérer paramsTree
    TTree* tree1 = (TTree*) f1->Get("paramsTree");
    if (!tree1) {
        std::cerr << "## ERREUR [step1] : TTree \"paramsTree\" introuvable dans Params.root\n";
        f1->Close();
        return;
    }
    std::cout << "[debug]    paramsTree trouvé : adresse = " << tree1 
              << " (entries = " << tree1->GetEntries() << ")\n";

    // 2) Ouvrir 100k_kaon_merged_digitized.root
    std::cout << "[debug] 2) Ouverture de 100k_kaon_merged_digitized.root...\n";
    TFile* f2 = TFile::Open("data/100k_kaon_merged_digitized.root", "READ");
    if (!f2 || f2->IsZombie()) {
        std::cerr << "## ERREUR [step2] : impossible d'ouvrir 100k_kaon_merged_digitized.root\n";
        f1->Close();
        return;
    }
    std::cout << "[debug]    100k_kaon_merged_digitized.root ouvert.\n";

    // Récupérer tree
    TTree* tree2 = (TTree*) f2->Get("tree");
    if (!tree2) {
        std::cerr << "## ERREUR [step2] : TTree \"tree\" introuvable dans 100k_kaon_merged_digitized.root\n";
        f1->Close();
        f2->Close();
        return;
    }
    std::cout << "[debug]    tree trouvé : adresse = " << tree2 
              << " (entries = " << tree2->GetEntries() << ")\n";


    // 2) Liaison des branches d'entrée
    std::vector<int>   *vThr = nullptr;
    std::vector<int>   *vK   = nullptr;
    // std::vector<float> *vx   = nullptr, *vy = nullptr, *vz = nullptr;
    // std::vector<int>   *vI   = nullptr, *vJ = nullptr;
    // Int_t particle1PDG = 0;

    // tree2->SetBranchAddress("particle1PDG", &particle1PDG);
    tree2->SetBranchAddress("thr", &vThr);
    tree2->SetBranchAddress("K",   &vK);
    // tree2->SetBranchAddress("x",   &vx);
    // tree2->SetBranchAddress("y",   &vy);
    // tree2->SetBranchAddress("z",   &vz);
    // tree2->SetBranchAddress("I",   &vI);
    // tree2->SetBranchAddress("J",   &vJ);

    // Déclaration des nouvelles branches
    Float_t Zbary_thr3 = 0.0f;
    Float_t Zbary_thr2 = 0.0f;
    Int_t N3 = 0;
    Int_t N2 = 0;
    TBranch* bZ3 = tree1->Branch("Zbary_thr3", &Zbary_thr3, "Zbary_thr3/F");
    TBranch* bZ2 = tree1->Branch("Zbary_thr2", &Zbary_thr2, "Zbary_thr2/F");
    TBranch* bN3 = tree1->Branch("N3", &N3, "N3/I");
    TBranch* bN2 = tree1->Branch("N2", &N2, "N2/I");

    // Boucle sur tous les événements
    Long64_t nEntries = tree2->GetEntries();
    std::cout << "[debug] Boucle sur " << nEntries << " événements...\n";
    for (Long64_t i = 0; i < nEntries; ++i) {
        tree2->GetEntry(i);
        tree1->GetEntry(i);

        // Calcul du barycentre pour seuil 3
        int    nHits3       = 0;
        double sumLayer3    = 0.0;
        // Calcul du barycentre pour seuil 2
        int    nHits2       = 0;
        double sumLayer2    = 0.0;

        for (size_t j = 0; j < vThr->size(); ++j) {
            int thr = vThr->at(j);
            int layer = vK->at(j);
            if (thr == 3) {
                nHits3++;
                sumLayer3 += static_cast<double>(layer);
            } else if (thr == 2) {
                nHits2++;
                sumLayer2 += static_cast<double>(layer);
            }
        }
        Zbary_thr3 = (nHits3 > 0) ? static_cast<Float_t>(sumLayer3 / nHits3) : 0.0f;
        Zbary_thr2 = (nHits2 > 0) ? static_cast<Float_t>(sumLayer2 / nHits2) : 0.0f;

        N3 = nHits3;
        N2 = nHits2;

        // Remplir les branches
        bZ3->Fill();
        bZ2->Fill();
        bN3->Fill();
        bN2->Fill();
    }

    // 4) Écrire et fermer
    f1->cd();
    tree1->Write("", TObject::kOverwrite);
    std::cout << "[debug] Branches ajoutées et Params.root mis à jour.\n";

    f2->Close();
    f1->Close();
    std::cout << "[debug] Fin de addParam()\n";
}



