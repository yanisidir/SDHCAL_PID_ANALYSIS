// extract.cpp
// Extract the shower parameters of every event of a digitized ROOT file.
//
// Build:
//     make features                             (from the repository root)
// or by hand, from the repository root:
//     g++ -std=c++17 `root-config --cflags` -I cpp/include -O2 -Wall -Wextra
//         <application and shared sources> -o analysis/1_shower_features/extract
//         `root-config --libs`
//
// Run:
//     ./extract <input.root> <treeName> <output.root>
//     ./extract                           (every particle under $SDHCAL_DATA_DIR)
//
// The absolute internship paths that used to be compiled into this file are gone:
// the default input and output directories come from $SDHCAL_DATA_DIR.

#include "RootShowerIO.h"
#include "ShowerFeatures.h"

#include <TFile.h>
#include <TSystem.h>
#include <TTree.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

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
    TFile* inputFileHandle = TFile::Open(inputFile, "READ");
    if (!inputFileHandle || inputFileHandle->IsZombie()) {
        std::cerr << "Impossible d'ouvrir " << inputFile << std::endl;
        return 1;
    }
    TTree* inputTree = dynamic_cast<TTree*>(inputFileHandle->Get(treeName));
    if (!inputTree) {
        std::cerr << "TTree '" << treeName << "' introuvable dans " << inputFile << std::endl;
        inputFileHandle->Close();
        return 1;
    }

    sdhcal::RootShowerInput input;
    sdhcal::RootInputOptions inputOptions;
    inputOptions.bindEnergy = true;
    sdhcal::bindShowerInput(*inputTree, input, inputOptions);
    if (!input.hasEventNumber) {
        std::cerr << "WARNING: no 'eventNumber' branch in the input; the output will carry the "
                     "row index instead. Two files produced from different simulations will then "
                     "collide on identifiers.\n";
    }

    TFile* outputFileHandle = TFile::Open(outputFile, "RECREATE");
    if (!outputFileHandle || outputFileHandle->IsZombie()) {
        std::cerr << "Impossible de créer " << outputFile << std::endl;
        inputFileHandle->Close();
        return 1;
    }
    TTree* outputTree = new TTree("paramsTree", "Paramètres de la gerbe");
    sdhcal::ShowerFeatures output;
    sdhcal::createParameterBranches(*outputTree, output, input.hasEnergy);

    const Long64_t nEntries = inputTree->GetEntries();
    for (Long64_t event = 0; event < nEntries; ++event) {
        inputTree->GetEntry(event);
        output = sdhcal::extractShowerFeatures(
            *input.thresholds, *input.layers, *input.x, *input.y, *input.z,
            *input.cellI, *input.cellJ, *input.time, input.particlePDG,
            input.hasEventNumber ? input.eventNumber : static_cast<int>(event),
            input.energy);
        outputTree->Fill();
    }

    outputFileHandle->Write();
    outputFileHandle->Close();
    inputFileHandle->Close();

    std::cout << "Analyse terminée, résultats dans " << outputFile << std::endl;
    return 0;
}
