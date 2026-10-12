#ifndef SDHCAL_SHOWER_FEATURES_H
#define SDHCAL_SHOWER_FEATURES_H

#include <vector>

namespace sdhcal {

// Flat event record written by the two feature-extraction applications.
// Native int/float members intentionally match the historical ROOT leaf types.
struct ShowerFeatures {
    int eventNumber = -1;
    float primaryEnergy = -1.0f;
    int particlePDG = 0;
    float Thr1 = 0.0f;
    float Thr2 = 0.0f;
    float Thr3 = 0.0f;
    int Begin = 48;
    float Radius = 0.0f;
    float Density = 0.0f;
    int NClusters = 0;
    float ratioThr23 = 0.0f;
    float Zbary = 0.0f;
    float Zrms = 0.0f;
    float PctHitsFirst10 = 0.0f;
    int PlanesWithClusmore2 = 0;
    float AvgClustSize = 0.0f;
    int MaxClustSize = 0;
    float lambda1 = 0.0f;
    float lambda2 = 0.0f;
    int N3 = 0;
    int N2 = 0;
    int N1 = 0;
    float tMin = 0.0f;
    float tMax = 0.0f;
    float tMean = 0.0f;
    float tSpread = 0.0f;
    float Nmax = 0.0f;
    float z0_fit = 0.0f;
    float Xmax = 0.0f;
    float lambda = 0.0f;
    int nTrackSegments = 0;
    float eccentricity3D = 0.0f;
    int nHitsTotal = 0;
    float sumThrTotal = 0.0f;
    int touchesLastLayer = 0;
};

// Run the historical ShowerAnalyzer once and copy its getters without changing
// formulas, floating-point order, selections, sentinels or fit handling.
ShowerFeatures extractShowerFeatures(
    const std::vector<int>& thresholds,
    const std::vector<int>& layers,
    const std::vector<float>& x,
    const std::vector<float>& y,
    const std::vector<float>& z,
    const std::vector<int>& cellI,
    const std::vector<int>& cellJ,
    const std::vector<float>& time,
    int particlePDG,
    int eventNumber,
    float primaryEnergy);

}  // namespace sdhcal

#endif  // SDHCAL_SHOWER_FEATURES_H
