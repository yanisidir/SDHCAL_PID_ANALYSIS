#include <TFile.h>
#include <TTree.h>

#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const char* outputPath = argv[1];
    const std::string mode = argv[2];
    const bool hasEventNumber = mode != "no-event-number";
    const bool hasEnergy = mode != "no-energy";
    const bool hasTime = mode != "missing-time";
    const bool wrongTimeType = mode == "wrong-time-type";

    TFile output(outputPath, "RECREATE");
    TTree tree("tree", "synthetic R9 fixture");

    int eventNumber = 0;
    int particle1PDG = 0;
    float energy = 0.0f;
    std::vector<int> thresholds, layers, cellI, cellJ, integerTime;
    std::vector<float> x, y, z, time;

    if (hasEventNumber) tree.Branch("eventNumber", &eventNumber, "eventNumber/I");
    tree.Branch("particle1PDG", &particle1PDG, "particle1PDG/I");
    if (hasEnergy) tree.Branch("energy", &energy, "energy/F");
    tree.Branch("thr", &thresholds);
    tree.Branch("K", &layers);
    tree.Branch("I", &cellI);
    tree.Branch("J", &cellJ);
    tree.Branch("x", &x);
    tree.Branch("y", &y);
    tree.Branch("z", &z);
    if (wrongTimeType) {
        tree.Branch("time", &integerTime);
    } else if (hasTime) {
        tree.Branch("time", &time);
    }

    const int identifiers[] = {101, 205, 309};
    const int particles[] = {-211, 2212, -211};
    const float energies[] = {5.0f, 10.0f, 30.0f};
    for (int event = 0; event < 3; ++event) {
        eventNumber = identifiers[event];
        particle1PDG = particles[event];
        energy = energies[event];
        thresholds.clear();
        layers.clear();
        cellI.clear();
        cellJ.clear();
        x.clear();
        y.clear();
        z.clear();
        time.clear();
        integerTime.clear();

        int hits = event == 0 ? 0 : (event == 1 ? 1 : 21);
        for (int index = 0; index < hits; ++index) {
            if (event == 1) {
                thresholds.push_back(1);
                layers.push_back(0);
                cellI.push_back(10);
                cellJ.push_back(20);
                x.push_back(1.0f);
                y.push_back(2.0f);
                z.push_back(3.0f);
                time.push_back(4.0f);
                integerTime.push_back(4);
            } else {
                const int repeatedThresholds[] = {1, 2, 3, 1};
                thresholds.push_back(index == 20 ? 3 : repeatedThresholds[index % 4]);
                layers.push_back(index == 20 ? 47 : index / 4);
                cellI.push_back(index == 20 ? 9 : index % 4);
                cellJ.push_back(index == 20 ? 9 : ((index % 4 == 1 || index % 4 == 3) ? 1 : 0));
                x.push_back(static_cast<float>(index));
                y.push_back(static_cast<float>(index % 4));
                z.push_back(index == 20 ? 470.0f : static_cast<float>(10 * (index / 4)));
                time.push_back(static_cast<float>(index) / 2.0f);
                integerTime.push_back(index / 2);
            }
        }
        tree.Fill();
    }

    tree.Write();
    output.Close();
    return 0;
}
