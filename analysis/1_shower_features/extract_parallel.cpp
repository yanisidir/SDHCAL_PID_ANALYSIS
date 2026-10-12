/**
 * extract_parallel.cpp
 * Conserve toutes les branches d'entrée et ajoute les paramètres calculés.
 *
 * Usage:
 *   ./extract_parallel input.root output.root [treeName]
 * Par défaut treeName="tree"
 *
 * Compilation (C++17), from the repository root :
 *   g++ -std=c++17 `root-config --cflags` -I cpp/include -Wall -Wextra -O2
 *       <application and shared sources> -o analysis/1_shower_features/extract_parallel
 *       `root-config --libs`
 *
 * Driven over many files by analysis/1_shower_features/parallel.sh, which takes
 * its directories from $SDHCAL_DATA_DIR rather than from absolute paths.
 */

#include "RootShowerIO.h"
#include "ShowerFeatures.h"

#include <TError.h>
#include <TFile.h>
#include <TSystem.h>
#include <TTree.h>

#include <iostream>
#include <string>

static int process_one(const char* inputFile,
                       const char* outputFile,
                       const char* treeName) {
    TFile* inputFileHandle = TFile::Open(inputFile, "READ");
    if (!inputFileHandle || inputFileHandle->IsZombie()) {
        ::Error("process_one", "Impossible d'ouvrir %s", inputFile);
        if (inputFileHandle) inputFileHandle->Close();
        return 1;
    }

    TTree* inputTree = dynamic_cast<TTree*>(inputFileHandle->Get(treeName));
    if (!inputTree) {
        ::Error("process_one", "TTree '%s' introuvable dans %s", treeName, inputFile);
        inputFileHandle->Close();
        return 1;
    }

    TFile* outputFileHandle = TFile::Open(outputFile, "RECREATE");
    if (!outputFileHandle || outputFileHandle->IsZombie()) {
        ::Error("process_one", "Impossible de créer %s", outputFile);
        inputFileHandle->Close();
        return 1;
    }
    outputFileHandle->cd();

    TTree* outputTree = inputTree->CloneTree(-1, "fast");
    if (!outputTree) {
        ::Error("process_one", "CloneTree a échoué pour %s", inputFile);
        outputFileHandle->Close();
        inputFileHandle->Close();
        return 1;
    }
    outputTree->SetName(treeName);
    outputTree->SetDirectory(outputFileHandle);

    sdhcal::RootShowerInput input;
    sdhcal::RootInputOptions inputOptions;
    inputOptions.disableUnusedBranches = true;
    sdhcal::bindShowerInput(*inputTree, input, inputOptions);
    if (!input.hasEventNumber) {
        std::cerr << "WARNING: no 'eventNumber' branch in the input; the output will carry "
                     "the row index instead.\n";
    }

    sdhcal::ShowerFeatures output;
    const sdhcal::EnrichmentBranches outputBranches =
        sdhcal::createEnrichmentBranches(*outputTree, output);

    const Long64_t nEntries = inputTree->GetEntries();
    for (Long64_t event = 0; event < nEntries; ++event) {
        inputTree->GetEntry(event);
        output = sdhcal::extractShowerFeatures(
            *input.thresholds, *input.layers, *input.x, *input.y, *input.z,
            *input.cellI, *input.cellJ, *input.time, input.particlePDG,
            input.hasEventNumber ? input.eventNumber : static_cast<int>(event),
            input.energy);
        outputBranches.fill();
    }

    outputFileHandle->cd();
    outputTree->Write();
    outputFileHandle->Close();
    inputFileHandle->Close();

    ::Info("process_one", "OK: %s -> %s", inputFile, outputFile);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " input.root output.root [treeName]\n";
        std::cerr << "Par defaut, treeName = \"tree\"\n";
        return 1;
    }
    const char* input = argv[1];
    const char* output = argv[2];
    const char* tree = (argc == 4 ? argv[3] : "tree");

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
