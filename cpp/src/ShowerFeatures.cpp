#include "ShowerFeatures.h"

#include "ShowerAnalyzer.h"

#include <cstddef>

namespace sdhcal {
namespace {

const int kLastLayer = 47;

}  // namespace

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
    float primaryEnergy) {
    ShowerAnalyzer analyzer(thresholds, layers, x, y, z, cellI, cellJ, time);
    analyzer.analyze();

    ShowerFeatures output;
    output.eventNumber = eventNumber;
    output.primaryEnergy = primaryEnergy;
    output.particlePDG = particlePDG;
    output.Thr1 = analyzer.getThr1();
    output.Thr2 = analyzer.getThr2();
    output.Thr3 = analyzer.getThr3();
    output.Begin = analyzer.getBegin();
    output.Radius = analyzer.getRadius();
    output.Density = analyzer.getDensity();
    output.NClusters = analyzer.getNClusters();
    output.ratioThr23 = analyzer.getRatioThr23();
    output.Zbary = analyzer.getZbary();
    output.Zrms = analyzer.getZrms();
    output.PctHitsFirst10 = analyzer.getPctHitsFirst10();
    output.PlanesWithClusmore2 = analyzer.getPlanesWithClusmore2();
    output.AvgClustSize = analyzer.getAvgClustSize();
    output.MaxClustSize = analyzer.getMaxClustSize();
    output.lambda1 = analyzer.getLambda1();
    output.lambda2 = analyzer.getLambda2();
    output.N3 = analyzer.getN3();
    output.N2 = analyzer.getN2();
    output.N1 = analyzer.getN1();
    output.tMin = analyzer.getTMin();
    output.tMax = analyzer.getTMax();
    output.tMean = analyzer.getTMean();
    output.tSpread = analyzer.getTSpread();
    output.Nmax = analyzer.getNmax();
    output.z0_fit = analyzer.getZ0Fit();
    output.Xmax = analyzer.getXmax();
    output.lambda = analyzer.getLambda();
    output.eccentricity3D = analyzer.getEccentricity3D();
    output.nTrackSegments = analyzer.getNTrackSegments();

    output.nHitsTotal = output.N1 + output.N2 + output.N3;
    output.sumThrTotal = static_cast<float>(output.N1)
                       + 2.0f * static_cast<float>(output.N2)
                       + 3.0f * static_cast<float>(output.N3);

    for (std::size_t index = 0; index < layers.size(); ++index) {
        if (layers[index] == kLastLayer) {
            output.touchesLastLayer = 1;
            break;
        }
    }
    return output;
}

}  // namespace sdhcal
