// computeParams.cpp
// Extract the shower parameters of every event of a digitized ROOT file.
//
// Build:
//     make shower-analyzer                      (from the repository root)
// or by hand:
//     g++ -std=c++17 `root-config --cflags` -I ShowerAnalyzer -O2 -Wall -Wextra \
//         ShowerAnalyzer/computeParams.cpp ShowerAnalyzer/ShowerAnalyzer.cpp \
//         -o ShowerAnalyzer/computeParams `root-config --libs`
//
// Run:
//     ./computeParams <input.root> <treeName> <output.root>
//     ./computeParams                           (every particle under $SDHCAL_DATA_DIR)
//
// The absolute internship paths that used to be compiled into this file are gone:
// the default input and output directories come from $SDHCAL_DATA_DIR.

#include "ShowerAnalyzer.h"
#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

int process(const char* inputFile, const char* treeName, const char* outputFile);

namespace {

/// Root of the ROOT inputs: $SDHCAL_DATA_DIR, or ./data.
std::string dataDir() {
    const char* fromEnv = std::getenv("SDHCAL_DATA_DIR");
    return (fromEnv && *fromEnv) ? std::string(fromEnv) : std::string("data");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 4) {
        return process(argv[1], argv[2], argv[3]);
    }
    if (argc != 1) {
        std::cerr << "usage: " << argv[0] << " [<input.root> <treeName> <output.root>]\n"
                  << "with no argument, every particle under $SDHCAL_DATA_DIR is processed.\n";
        return 2;
    }

    const std::string base = dataDir();
    const char* particles[] = {"pi", "kaon", "proton"};

    std::vector<std::pair<std::string, std::string> > files;
    for (const char* particle : particles) {
        files.push_back(std::make_pair(
            base + "/digitized/130k_" + particle + "_E1to130_digitized.root",
            base + "/params/130k_" + particle + "_E1to130_params.root"));
    }

    if (gSystem) gSystem->mkdir((base + "/params").c_str(), kTRUE);

    int exit_code = 0;
    for (size_t i = 0; i < files.size(); ++i) {
        std::cout << "\n>>> Processing " << files[i].first << std::endl;
        exit_code |= process(files[i].first.c_str(), "tree", files[i].second.c_str());
    }
    return exit_code;
}

int process(const char* inputFile, const char* treeName, const char* outputFile) {
    // 1) Fichier d'entrée
    TFile* f_in = TFile::Open(inputFile, "READ");
    if (!f_in || f_in->IsZombie()) {
        std::cerr << "Impossible d'ouvrir " << inputFile << std::endl;
        return 1;
    }
    TTree* tree = dynamic_cast<TTree*>(f_in->Get(treeName));
    if (!tree) {
        std::cerr << "TTree '" << treeName << "' introuvable dans " << inputFile << std::endl;
        f_in->Close();
        return 1;
    }

    // 2) Branches d'entrée
    std::vector<int>*   vThr  = 0;
    std::vector<int>*   vK    = 0;
    std::vector<float>* vx    = 0;
    std::vector<float>* vy    = 0;
    std::vector<float>* vz    = 0;
    std::vector<int>*   vI    = 0;
    std::vector<int>*   vJ    = 0;
    std::vector<float>* vTime = 0;
    Int_t particle1PDG = 0;
    Int_t inEventNumber = -1;

    // An event identifier is needed downstream to prove that a held-out sample
    // shares no event with the training files. When the input tree carries one we
    // propagate it; otherwise we fall back to the row index, which is only
    // meaningful as long as the file is not reshuffled.
    const bool hasEventNumber =
        tree->GetListOfBranches()->FindObject("eventNumber") != nullptr;
    if (hasEventNumber) {
        tree->SetBranchAddress("eventNumber", &inEventNumber);
    } else {
        std::cerr << "WARNING: no 'eventNumber' branch in the input; the output will carry the "
                     "row index instead. Two files produced from different simulations will then "
                     "collide on identifiers.\n";
    }

    tree->SetBranchAddress("particle1PDG", &particle1PDG);
    tree->SetBranchAddress("thr",         &vThr);
    tree->SetBranchAddress("K",           &vK);
    tree->SetBranchAddress("x",           &vx);
    tree->SetBranchAddress("y",           &vy);
    tree->SetBranchAddress("z",           &vz);
    tree->SetBranchAddress("I",           &vI);
    tree->SetBranchAddress("J",           &vJ);
    tree->SetBranchAddress("time",        &vTime);

    // 3) Fichier/Tree de sortie
    TFile* f_out = TFile::Open(outputFile, "RECREATE");
    if (!f_out || f_out->IsZombie()) {
        std::cerr << "Impossible de créer " << outputFile << std::endl;
        f_in->Close();
        return 1;
    }
    TTree* outTree = new TTree("paramsTree", "Paramètres de la gerbe");

    // Variables de sortie
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
    Int_t   eventNumber;
    Int_t   touchesLastLayer;

    // Branches de sortie
    outTree->Branch("eventNumber",        &eventNumber,        "eventNumber/I");
    outTree->Branch("particlePDG",        &particlePDG,        "particlePDG/I");
    outTree->Branch("Thr1",               &Thr1,               "Thr1/F");
    outTree->Branch("Thr2",               &Thr2,               "Thr2/F");
    outTree->Branch("Thr3",               &Thr3,               "Thr3/F");
    // Begin is the first layer of the first quadruplet of consecutive layers with
    // at least 4 hits each. When no such quadruplet exists the value is the
    // sentinel 48, one past the last layer, which is outside the physical range
    // on purpose: it is a "no shower start found" marker, not a depth. Any model
    // using Begin as a continuous variable sees it as an extreme value.
    outTree->Branch("Begin",              &Begin,              "Begin/I");
    outTree->Branch("Radius",             &Radius,             "Radius/F");
    outTree->Branch("Density",            &Density,            "Density/F");
    outTree->Branch("NClusters",          &NClusters,          "NClusters/I");
    outTree->Branch("ratioThr23",         &ratioThr23,         "ratioThr23/F");
    outTree->Branch("Zbary",              &Zbary,              "Zbary/F");
    outTree->Branch("Zrms",               &Zrms,               "Zrms/F");
    outTree->Branch("PctHitsFirst10",     &PctHitsFirst10,     "PctHitsFirst10/F");
    outTree->Branch("PlanesWithClusmore2",&PlanesWithClusmore2,"PlanesWithClusmore2/I");
    outTree->Branch("AvgClustSize",       &AvgClustSize,       "AvgClustSize/F");
    outTree->Branch("MaxClustSize",       &MaxClustSize,       "MaxClustSize/I");
    outTree->Branch("lambda1",            &lambda1,            "lambda1/F");
    outTree->Branch("lambda2",            &lambda2,            "lambda2/F");
    outTree->Branch("N3",                 &N3,                 "N3/I");
    outTree->Branch("N2",                 &N2,                 "N2/I");
    outTree->Branch("N1",                 &N1,                 "N1/I");
    outTree->Branch("tMin",               &tMin,               "tMin/F");
    outTree->Branch("tMax",               &tMax,               "tMax/F");
    outTree->Branch("tMean",              &tMean,              "tMean/F");
    outTree->Branch("tSpread",            &tSpread,            "tSpread/F");
    outTree->Branch("Nmax",               &Nmax,               "Nmax/F");
    outTree->Branch("z0_fit",             &z0_fit,             "z0_fit/F");
    outTree->Branch("Xmax",               &Xmax,               "Xmax/F");
    outTree->Branch("lambda",             &lambda,             "lambda/F");
    outTree->Branch("nTrackSegments",     &nTrackSegments,     "nTrackSegments/I");
    outTree->Branch("eccentricity3D",     &eccentricity3D,     "eccentricity3D/F");
    outTree->Branch("nHitsTotal",         &nHitsTotal,         "nHitsTotal/I");
    outTree->Branch("sumThrTotal",        &sumThrTotal,        "sumThrTotal/F");
    // 1 when the shower reaches the last active layer, i.e. when it is not
    // longitudinally contained. The chi2 chain rejects those events; exposing the
    // flag lets the machine-learning chain apply exactly the same selection, so
    // that the two methods are compared on the same sample.
    outTree->Branch("touchesLastLayer",   &touchesLastLayer,   "touchesLastLayer/I");

    // 4) Boucle événements
    const Int_t kLastLayer = 47;  // index of the last active layer of the SDHCAL
    const Long64_t nEntries = tree->GetEntries();
    for (Long64_t ev = 0; ev < nEntries; ++ev) {
        tree->GetEntry(ev);

        ShowerAnalyzer analyzer(*vThr, *vK, *vx, *vy, *vz, *vI, *vJ, *vTime);
        analyzer.analyze();

        eventNumber         = hasEventNumber ? inEventNumber : static_cast<Int_t>(ev);

        touchesLastLayer = 0;
        for (std::size_t h = 0; h < vK->size(); ++h) {
            if ((*vK)[h] == kLastLayer) { touchesLastLayer = 1; break; }
        }

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

        outTree->Fill();
    }

    // 5) Sauvegarde
    f_out->Write();
    f_out->Close();
    f_in->Close();

    std::cout << "Analyse terminée, résultats dans " << outputFile << std::endl;
    return 0;
}
