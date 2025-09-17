/**
 * computeParams_parallel.cpp
 * Conserve toutes les branches d'entrée et ajoute les paramètres calculés.
 *
 * Usage:
 *   ./computeParams_parallel input.root output.root [treeName]
 * Par défaut treeName="tree"
 *
 * Compilation (C++11) :
 *   g++ -std=c++11 `root-config --cflags` -I/gridgroup/ilc/midir/analyse/ShowerAnalyzer -Wall -Wextra -O2 -c computeParams_parallel.cpp -o computeParams_parallel.o
 *   g++ -std=c++11 `root-config --cflags` -Wall -Wextra -O2 -c /gridgroup/ilc/midir/analyse/ShowerAnalyzer/ShowerAnalyzer.cpp -o ShowerAnalyzer.o
 *   g++ computeParams_parallel.o ShowerAnalyzer.o -o /gridgroup/ilc/midir/analyse/ShowerAnalyzer/computeParams_parallel `root-config --libs`
 */

#include "/gridgroup/ilc/midir/analyse/ShowerAnalyzer/ShowerAnalyzer.h"

#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>
#include <TError.h>

#include <iostream>
#include <vector>
#include <string>

static int process_one(const char* inputFile,
                       const char* outputFile,
                       const char* treeName)
{
    // 1) Ouverture entrée
    TFile* fin = TFile::Open(inputFile, "READ");
    if (!fin || fin->IsZombie()) {
        ::Error("process_one", "Impossible d'ouvrir %s", inputFile);
        if (fin) fin->Close();
        return 1;
    }

    TTree* tin = dynamic_cast<TTree*>(fin->Get(treeName));
    if (!tin) {
        ::Error("process_one", "TTree '%s' introuvable dans %s", treeName, inputFile);
        fin->Close();
        return 1;
    }

    // 2) Prépare la sortie et clone TOUT l'arbre (entrées + branches)
    TFile* fout = TFile::Open(outputFile, "RECREATE");
    if (!fout || fout->IsZombie()) {
        ::Error("process_one", "Impossible de créer %s", outputFile);
        fin->Close();
        return 1;
    }
    fout->cd();

    // Clone rapide : recopie immédiatement toutes les entrées/branches existantes
    TTree* tout = tin->CloneTree(-1, "fast");
    if (!tout) {
        ::Error("process_one", "CloneTree a échoué pour %s", inputFile);
        fout->Close();
        fin->Close();
        return 1;
    }
    tout->SetName(treeName);
    tout->SetDirectory(fout);

    // 3) Branches d'entrée nécessaires à ShowerAnalyzer (lire seulement l’utile)
    std::vector<int>*   vThr  = nullptr;
    std::vector<int>*   vK    = nullptr;
    std::vector<float>* vx    = nullptr;
    std::vector<float>* vy    = nullptr;
    std::vector<float>* vz    = nullptr;
    std::vector<int>*   vI    = nullptr;
    std::vector<int>*   vJ    = nullptr;
    std::vector<float>* vTime = nullptr;
    Int_t               particle1PDG = 0;

    tin->SetBranchStatus("*", 0);
    tin->SetBranchStatus("particle1PDG", 1);
    tin->SetBranchStatus("thr", 1);
    tin->SetBranchStatus("K", 1);
    tin->SetBranchStatus("x", 1);
    tin->SetBranchStatus("y", 1);
    tin->SetBranchStatus("z", 1);
    tin->SetBranchStatus("I", 1);
    tin->SetBranchStatus("J", 1);
    tin->SetBranchStatus("time", 1);

    tin->SetBranchAddress("particle1PDG", &particle1PDG);
    tin->SetBranchAddress("thr",         &vThr);
    tin->SetBranchAddress("K",           &vK);
    tin->SetBranchAddress("x",           &vx);
    tin->SetBranchAddress("y",           &vy);
    tin->SetBranchAddress("z",           &vz);
    tin->SetBranchAddress("I",           &vI);
    tin->SetBranchAddress("J",           &vJ);
    tin->SetBranchAddress("time",        &vTime);

    // 4) Nouvelles branches (variables + création des branches)
    Int_t   particlePDG;
    Float_t Thr1, Thr2, Thr3;
    Int_t   Begin;
    Float_t Radius, Density;
    Int_t   NClusters;
    Float_t ratioThr23;
    Float_t Zbary, Zrms;
    Float_t PctHitsFirst10;
    Int_t   PlanesWithClusmore2;
    Float_t AvgClustSize;
    Int_t   MaxClustSize;
    Float_t lambda1, lambda2;
    Int_t   N3, N2, N1;
    Float_t tMin, tMax, tMean, tSpread;
    Float_t Nmax, z0_fit, Xmax, lambda;
    Int_t   nTrackSegments;
    Float_t eccentricity3D;
    Int_t   nHitsTotal;
    Float_t sumThrTotal;

    // On garde les pointeurs de TBranch* pour Fill() sans appeler tout->Fill()
    TBranch* b_particlePDG         = tout->Branch("particlePDG",        &particlePDG,        "particlePDG/I");
    TBranch* b_Thr1                = tout->Branch("Thr1",               &Thr1,               "Thr1/F");
    TBranch* b_Thr2                = tout->Branch("Thr2",               &Thr2,               "Thr2/F");
    TBranch* b_Thr3                = tout->Branch("Thr3",               &Thr3,               "Thr3/F");
    TBranch* b_Begin               = tout->Branch("Begin",              &Begin,              "Begin/I");
    TBranch* b_Radius              = tout->Branch("Radius",             &Radius,             "Radius/F");
    TBranch* b_Density             = tout->Branch("Density",            &Density,            "Density/F");
    TBranch* b_NClusters           = tout->Branch("NClusters",          &NClusters,          "NClusters/I");
    TBranch* b_ratioThr23          = tout->Branch("ratioThr23",         &ratioThr23,         "ratioThr23/F");
    TBranch* b_Zbary               = tout->Branch("Zbary",              &Zbary,              "Zbary/F");
    TBranch* b_Zrms                = tout->Branch("Zrms",               &Zrms,               "Zrms/F");
    TBranch* b_PctHitsFirst10      = tout->Branch("PctHitsFirst10",     &PctHitsFirst10,     "PctHitsFirst10/F");
    TBranch* b_PlanesWithClusmore2 = tout->Branch("PlanesWithClusmore2",&PlanesWithClusmore2,"PlanesWithClusmore2/I");
    TBranch* b_AvgClustSize        = tout->Branch("AvgClustSize",       &AvgClustSize,       "AvgClustSize/F");
    TBranch* b_MaxClustSize        = tout->Branch("MaxClustSize",       &MaxClustSize,       "MaxClustSize/I");
    TBranch* b_lambda1             = tout->Branch("lambda1",            &lambda1,            "lambda1/F");
    TBranch* b_lambda2             = tout->Branch("lambda2",            &lambda2,            "lambda2/F");
    TBranch* b_N3                  = tout->Branch("N3",                 &N3,                 "N3/I");
    TBranch* b_N2                  = tout->Branch("N2",                 &N2,                 "N2/I");
    TBranch* b_N1                  = tout->Branch("N1",                 &N1,                 "N1/I");
    TBranch* b_tMin                = tout->Branch("tMin",               &tMin,               "tMin/F");
    TBranch* b_tMax                = tout->Branch("tMax",               &tMax,               "tMax/F");
    TBranch* b_tMean               = tout->Branch("tMean",              &tMean,              "tMean/F");
    TBranch* b_tSpread             = tout->Branch("tSpread",            &tSpread,            "tSpread/F");
    TBranch* b_Nmax                = tout->Branch("Nmax",               &Nmax,               "Nmax/F");
    TBranch* b_z0_fit              = tout->Branch("z0_fit",             &z0_fit,             "z0_fit/F");
    TBranch* b_Xmax                = tout->Branch("Xmax",               &Xmax,               "Xmax/F");
    TBranch* b_lambda              = tout->Branch("lambda",             &lambda,             "lambda/F");
    TBranch* b_nTrackSegments      = tout->Branch("nTrackSegments",     &nTrackSegments,     "nTrackSegments/I");
    TBranch* b_eccentricity3D      = tout->Branch("eccentricity3D",     &eccentricity3D,     "eccentricity3D/F");
    TBranch* b_nHitsTotal          = tout->Branch("nHitsTotal",         &nHitsTotal,         "nHitsTotal/I");
    TBranch* b_sumThrTotal         = tout->Branch("sumThrTotal",        &sumThrTotal,        "sumThrTotal/F");

    // 5) Boucle événements : on lit l'entrée, on calcule,
    //    puis on remplit UNIQUEMENT les nouvelles branches (pas de tout->Fill()).
    const Long64_t nEntries = tin->GetEntries();
    for (Long64_t ev = 0; ev < nEntries; ++ev) {
        tin->GetEntry(ev);

        // Calcul via ShowerAnalyzer
        ShowerAnalyzer analyzer(*vThr, *vK, *vx, *vy, *vz, *vI, *vJ, *vTime);
        analyzer.analyze();

        particlePDG         = particle1PDG;
        Thr1                = analyzer.getThr1();
        Thr2                = analyzer.getThr2();
        Thr3                = analyzer.getThr3();
        Begin               = analyzer.getBegin();
        Radius              = analyzer.getRadius();
        Density             = analyzer.getDensity();
        NClusters           = analyzer.getNClusters();
        ratioThr23          = analyzer.getRatioThr23();
        Zbary               = analyzer.getZbary();
        Zrms                = analyzer.getZrms();
        PctHitsFirst10      = analyzer.getPctHitsFirst10();
        PlanesWithClusmore2 = analyzer.getPlanesWithClusmore2();
        AvgClustSize        = analyzer.getAvgClustSize();
        MaxClustSize        = analyzer.getMaxClustSize();
        lambda1             = analyzer.getLambda1();
        lambda2             = analyzer.getLambda2();
        N3                  = analyzer.getN3();
        N2                  = analyzer.getN2();
        N1                  = analyzer.getN1();
        tMin                = analyzer.getTMin();
        tMax                = analyzer.getTMax();
        tMean               = analyzer.getTMean();
        tSpread             = analyzer.getTSpread();
        Nmax                = analyzer.getNmax();
        z0_fit              = analyzer.getZ0Fit();
        Xmax                = analyzer.getXmax();
        lambda              = analyzer.getLambda();
        eccentricity3D      = analyzer.getEccentricity3D();
        nTrackSegments      = analyzer.getNTrackSegments();

        nHitsTotal          = N1 + N2 + N3;
        sumThrTotal         = static_cast<float>(N1) + 2.0f*static_cast<float>(N2) + 3.0f*static_cast<float>(N3);

        // Remplir uniquement les nouvelles branches pour CETTE entrée
        b_particlePDG->Fill();
        b_Thr1->Fill();             b_Thr2->Fill();            b_Thr3->Fill();
        b_Begin->Fill();            b_Radius->Fill();          b_Density->Fill();
        b_NClusters->Fill();        b_ratioThr23->Fill();
        b_Zbary->Fill();            b_Zrms->Fill();
        b_PctHitsFirst10->Fill();   b_PlanesWithClusmore2->Fill();
        b_AvgClustSize->Fill();     b_MaxClustSize->Fill();
        b_lambda1->Fill();          b_lambda2->Fill();
        b_N3->Fill();               b_N2->Fill();              b_N1->Fill();
        b_tMin->Fill();             b_tMax->Fill();            b_tMean->Fill();   b_tSpread->Fill();
        b_Nmax->Fill();             b_z0_fit->Fill();          b_Xmax->Fill();    b_lambda->Fill();
        b_nTrackSegments->Fill();   b_eccentricity3D->Fill();
        b_nHitsTotal->Fill();       b_sumThrTotal->Fill();
    }

    // 6) Sauvegarde
    fout->cd();
    tout->Write();    // écrit l'arbre enrichi (anciennes + nouvelles branches)
    fout->Close();
    fin->Close();

    ::Info("process_one", "OK: %s -> %s", inputFile, outputFile);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " input.root output.root [treeName]\n";
        std::cerr << "Par defaut, treeName = \"tree\"\n";
        return 1;
    }
    const char* input  = argv[1];
    const char* output = argv[2];
    const char* tree   = (argc == 4 ? argv[3] : "tree");

    // Crée le dossier de sortie si nécessaire
    if (gSystem) {
        std::string outPath(output);
        std::string dir = outPath;
        std::size_t pos = dir.find_last_of('/');
        if (pos != std::string::npos) {
            dir = dir.substr(0, pos);
            if (!dir.empty()) gSystem->mkdir(dir.c_str(), kTRUE);
        }
    }

    return process_one(input, output, tree);
}
