#include "ShowerFeatures.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void close(float actual, float expected, float tolerance, const std::string& name) {
    require(std::fabs(actual - expected) <= tolerance,
            name + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual));
}

sdhcal::ShowerFeatures extract(
    const std::vector<int>& thresholds,
    const std::vector<int>& layers,
    const std::vector<int>& cellI,
    const std::vector<int>& cellJ,
    const std::vector<float>& x,
    const std::vector<float>& y,
    const std::vector<float>& z,
    const std::vector<float>& time,
    int eventNumber) {
    return sdhcal::extractShowerFeatures(
        thresholds, layers, x, y, z, cellI, cellJ, time,
        -211, eventNumber, 30.0f);
}

}  // namespace

int main() {
    const std::vector<int> noInts;
    const std::vector<float> noFloats;
    const sdhcal::ShowerFeatures empty =
        extract(noInts, noInts, noInts, noInts, noFloats, noFloats, noFloats, noFloats, 101);
    require(empty.eventNumber == 101, "empty event identifier");
    require(empty.Begin == 48, "empty Begin sentinel");
    require(empty.nHitsTotal == 0, "empty hit count");
    require(std::isnan(empty.Nmax), "empty Gaisser-Hillas Nmax");
    close(empty.tMean, 0.0f, 0.0f, "empty timing");

    const sdhcal::ShowerFeatures single = extract(
        {1}, {0}, {10}, {20}, {1.0f}, {2.0f}, {3.0f}, {4.0f}, 205);
    require(single.N1 == 1 && single.N2 == 0 && single.N3 == 0, "single-hit thresholds");
    close(single.Thr1, 100.0f, 0.0f, "single-hit Thr1");
    close(single.Density, 1.0f, 0.0f, "single-hit density");
    close(single.tMin, 4.0f, 0.0f, "single-hit tMin");
    close(single.tMax, 4.0f, 0.0f, "single-hit tMax");

    std::vector<int> thresholds;
    std::vector<int> layers;
    std::vector<int> cellI;
    std::vector<int> cellJ;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    std::vector<float> time;
    for (int index = 0; index < 20; ++index) {
        const int repeatedThresholds[] = {1, 2, 3, 1};
        thresholds.push_back(repeatedThresholds[index % 4]);
        layers.push_back(index / 4);
        cellI.push_back(index % 4);
        cellJ.push_back(index % 4 == 1 || index % 4 == 3 ? 1 : 0);
        x.push_back(static_cast<float>(index));
        y.push_back(static_cast<float>(index % 4));
        z.push_back(static_cast<float>(10 * (index / 4)));
        time.push_back(static_cast<float>(index) / 2.0f);
    }
    thresholds.push_back(3);
    layers.push_back(47);
    cellI.push_back(9);
    cellJ.push_back(9);
    x.push_back(20.0f);
    y.push_back(0.0f);
    z.push_back(470.0f);
    time.push_back(10.0f);

    const sdhcal::ShowerFeatures multiple =
        extract(thresholds, layers, cellI, cellJ, x, y, z, time, 309);
    require(multiple.Begin == 0, "four populated consecutive layers");
    require(multiple.N1 == 10 && multiple.N2 == 5 && multiple.N3 == 6, "threshold counts");
    require(multiple.nHitsTotal == 21, "multiple-event hit count");
    require(multiple.touchesLastLayer == 1, "last-layer flag");
    close(multiple.sumThrTotal, 38.0f, 0.0f, "threshold sum");
    close(multiple.Zbary, 4.142857074737549f, 1.0e-6f, "longitudinal barycenter");
    close(multiple.Zrms, 9.682019233703613f, 1.0e-6f, "longitudinal RMS");
    close(multiple.Radius, 5.395670413970947f, 1.0e-5f, "transverse radius");
    close(multiple.Density, 2.7321274280548096f, 1.0e-6f, "density");
    close(multiple.lambda1, 36.67307662963867f, 1.0e-5f, "lambda1");
    close(multiple.lambda2, 1.2861078977584839f, 1.0e-6f, "lambda2");
    close(multiple.eccentricity3D, 0.9969367384910583f, 1.0e-6f, "eccentricity3D");
    close(multiple.Nmax, 7.088703632354736f, 1.0e-4f, "Gaisser-Hillas Nmax");
    close(multiple.z0_fit, -79.98625946044922f, 1.0e-3f, "Gaisser-Hillas z0");
    close(multiple.Xmax, 18.466054916381836f, 1.0e-4f, "Gaisser-Hillas Xmax");
    close(multiple.lambda, 79.99503326416016f, 1.0e-3f, "Gaisser-Hillas lambda");

    std::cout << "ShowerAnalyzer characterization: OK\n";
    return 0;
}
