#include "RootShowerIO.h"

#include <TBranch.h>
#include <TObjArray.h>
#include <TTree.h>

namespace sdhcal {

void bindShowerInput(TTree& tree, RootShowerInput& input, RootInputOptions options) {
    if (options.disableUnusedBranches) {
        tree.SetBranchStatus("*", 0);
        tree.SetBranchStatus("particle1PDG", 1);
        tree.SetBranchStatus("thr", 1);
        tree.SetBranchStatus("K", 1);
        tree.SetBranchStatus("x", 1);
        tree.SetBranchStatus("y", 1);
        tree.SetBranchStatus("z", 1);
        tree.SetBranchStatus("I", 1);
        tree.SetBranchStatus("J", 1);
        tree.SetBranchStatus("time", 1);
    }

    input.hasEventNumber = tree.GetListOfBranches()->FindObject("eventNumber") != nullptr;
    if (input.hasEventNumber) {
        if (options.disableUnusedBranches) tree.SetBranchStatus("eventNumber", 1);
        tree.SetBranchAddress("eventNumber", &input.eventNumber);
    }

    input.hasEnergy = options.bindEnergy
                   && tree.GetListOfBranches()->FindObject("energy") != nullptr;
    if (input.hasEnergy) {
        if (options.disableUnusedBranches) tree.SetBranchStatus("energy", 1);
        tree.SetBranchAddress("energy", &input.energy);
    }

    tree.SetBranchAddress("particle1PDG", &input.particlePDG);
    tree.SetBranchAddress("thr", &input.thresholds);
    tree.SetBranchAddress("K", &input.layers);
    tree.SetBranchAddress("x", &input.x);
    tree.SetBranchAddress("y", &input.y);
    tree.SetBranchAddress("z", &input.z);
    tree.SetBranchAddress("I", &input.cellI);
    tree.SetBranchAddress("J", &input.cellJ);
    tree.SetBranchAddress("time", &input.time);
}

void createParameterBranches(TTree& tree, ShowerFeatures& output, bool includePrimaryEnergy) {
    tree.Branch("eventNumber", &output.eventNumber, "eventNumber/I");
    if (includePrimaryEnergy) {
        tree.Branch("primaryEnergy", &output.primaryEnergy, "primaryEnergy/F");
    }
    tree.Branch("particlePDG", &output.particlePDG, "particlePDG/I");
    tree.Branch("Thr1", &output.Thr1, "Thr1/F");
    tree.Branch("Thr2", &output.Thr2, "Thr2/F");
    tree.Branch("Thr3", &output.Thr3, "Thr3/F");
    tree.Branch("Begin", &output.Begin, "Begin/I");
    tree.Branch("Radius", &output.Radius, "Radius/F");
    tree.Branch("Density", &output.Density, "Density/F");
    tree.Branch("NClusters", &output.NClusters, "NClusters/I");
    tree.Branch("ratioThr23", &output.ratioThr23, "ratioThr23/F");
    tree.Branch("Zbary", &output.Zbary, "Zbary/F");
    tree.Branch("Zrms", &output.Zrms, "Zrms/F");
    tree.Branch("PctHitsFirst10", &output.PctHitsFirst10, "PctHitsFirst10/F");
    tree.Branch("PlanesWithClusmore2", &output.PlanesWithClusmore2, "PlanesWithClusmore2/I");
    tree.Branch("AvgClustSize", &output.AvgClustSize, "AvgClustSize/F");
    tree.Branch("MaxClustSize", &output.MaxClustSize, "MaxClustSize/I");
    tree.Branch("lambda1", &output.lambda1, "lambda1/F");
    tree.Branch("lambda2", &output.lambda2, "lambda2/F");
    tree.Branch("N3", &output.N3, "N3/I");
    tree.Branch("N2", &output.N2, "N2/I");
    tree.Branch("N1", &output.N1, "N1/I");
    tree.Branch("tMin", &output.tMin, "tMin/F");
    tree.Branch("tMax", &output.tMax, "tMax/F");
    tree.Branch("tMean", &output.tMean, "tMean/F");
    tree.Branch("tSpread", &output.tSpread, "tSpread/F");
    tree.Branch("Nmax", &output.Nmax, "Nmax/F");
    tree.Branch("z0_fit", &output.z0_fit, "z0_fit/F");
    tree.Branch("Xmax", &output.Xmax, "Xmax/F");
    tree.Branch("lambda", &output.lambda, "lambda/F");
    tree.Branch("nTrackSegments", &output.nTrackSegments, "nTrackSegments/I");
    tree.Branch("eccentricity3D", &output.eccentricity3D, "eccentricity3D/F");
    tree.Branch("nHitsTotal", &output.nHitsTotal, "nHitsTotal/I");
    tree.Branch("sumThrTotal", &output.sumThrTotal, "sumThrTotal/F");
    tree.Branch("touchesLastLayer", &output.touchesLastLayer, "touchesLastLayer/I");
}

EnrichmentBranches createEnrichmentBranches(TTree& tree, ShowerFeatures& output) {
    EnrichmentBranches result;
    result.branches_.push_back(tree.Branch("eventNumber", &output.eventNumber, "eventNumber/I"));
    result.branches_.push_back(
        tree.Branch("touchesLastLayer", &output.touchesLastLayer, "touchesLastLayer/I"));
    result.branches_.push_back(tree.Branch("particlePDG", &output.particlePDG, "particlePDG/I"));
    result.branches_.push_back(tree.Branch("Thr1", &output.Thr1, "Thr1/F"));
    result.branches_.push_back(tree.Branch("Thr2", &output.Thr2, "Thr2/F"));
    result.branches_.push_back(tree.Branch("Thr3", &output.Thr3, "Thr3/F"));
    result.branches_.push_back(tree.Branch("Begin", &output.Begin, "Begin/I"));
    result.branches_.push_back(tree.Branch("Radius", &output.Radius, "Radius/F"));
    result.branches_.push_back(tree.Branch("Density", &output.Density, "Density/F"));
    result.branches_.push_back(tree.Branch("NClusters", &output.NClusters, "NClusters/I"));
    result.branches_.push_back(tree.Branch("ratioThr23", &output.ratioThr23, "ratioThr23/F"));
    result.branches_.push_back(tree.Branch("Zbary", &output.Zbary, "Zbary/F"));
    result.branches_.push_back(tree.Branch("Zrms", &output.Zrms, "Zrms/F"));
    result.branches_.push_back(
        tree.Branch("PctHitsFirst10", &output.PctHitsFirst10, "PctHitsFirst10/F"));
    result.branches_.push_back(tree.Branch(
        "PlanesWithClusmore2", &output.PlanesWithClusmore2, "PlanesWithClusmore2/I"));
    result.branches_.push_back(
        tree.Branch("AvgClustSize", &output.AvgClustSize, "AvgClustSize/F"));
    result.branches_.push_back(
        tree.Branch("MaxClustSize", &output.MaxClustSize, "MaxClustSize/I"));
    result.branches_.push_back(tree.Branch("lambda1", &output.lambda1, "lambda1/F"));
    result.branches_.push_back(tree.Branch("lambda2", &output.lambda2, "lambda2/F"));
    result.branches_.push_back(tree.Branch("N3", &output.N3, "N3/I"));
    result.branches_.push_back(tree.Branch("N2", &output.N2, "N2/I"));
    result.branches_.push_back(tree.Branch("N1", &output.N1, "N1/I"));
    result.branches_.push_back(tree.Branch("tMin", &output.tMin, "tMin/F"));
    result.branches_.push_back(tree.Branch("tMax", &output.tMax, "tMax/F"));
    result.branches_.push_back(tree.Branch("tMean", &output.tMean, "tMean/F"));
    result.branches_.push_back(tree.Branch("tSpread", &output.tSpread, "tSpread/F"));
    result.branches_.push_back(tree.Branch("Nmax", &output.Nmax, "Nmax/F"));
    result.branches_.push_back(tree.Branch("z0_fit", &output.z0_fit, "z0_fit/F"));
    result.branches_.push_back(tree.Branch("Xmax", &output.Xmax, "Xmax/F"));
    result.branches_.push_back(tree.Branch("lambda", &output.lambda, "lambda/F"));
    result.branches_.push_back(
        tree.Branch("nTrackSegments", &output.nTrackSegments, "nTrackSegments/I"));
    result.branches_.push_back(
        tree.Branch("eccentricity3D", &output.eccentricity3D, "eccentricity3D/F"));
    result.branches_.push_back(tree.Branch("nHitsTotal", &output.nHitsTotal, "nHitsTotal/I"));
    result.branches_.push_back(
        tree.Branch("sumThrTotal", &output.sumThrTotal, "sumThrTotal/F"));
    return result;
}

void EnrichmentBranches::fill() const {
    for (TBranch* branch : branches_) branch->Fill();
}

}  // namespace sdhcal
