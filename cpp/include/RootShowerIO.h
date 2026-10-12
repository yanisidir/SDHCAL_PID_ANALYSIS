#ifndef SDHCAL_ROOT_SHOWER_IO_H
#define SDHCAL_ROOT_SHOWER_IO_H

#include "ShowerFeatures.h"

#include <vector>

class TBranch;
class TTree;

namespace sdhcal {

// Addresses owned by ROOT and rebound for each input entry. The pointed-to hit
// vectors remain owned by the input TTree.
struct RootShowerInput {
    std::vector<int>* thresholds = nullptr;
    std::vector<int>* layers = nullptr;
    std::vector<float>* x = nullptr;
    std::vector<float>* y = nullptr;
    std::vector<float>* z = nullptr;
    std::vector<int>* cellI = nullptr;
    std::vector<int>* cellJ = nullptr;
    std::vector<float>* time = nullptr;
    int particlePDG = 0;
    int eventNumber = -1;
    float energy = -1.0f;
    bool hasEventNumber = false;
    bool hasEnergy = false;
};

struct RootInputOptions {
    bool disableUnusedBranches = false;
    bool bindEnergy = false;
};

// Reproduce the historical input names and address binding. Optional branches
// are reported through RootShowerInput and handled by each application.
void bindShowerInput(TTree& tree, RootShowerInput& input, RootInputOptions options = {});

// Create the paramsTree schema in its historical branch order.
void createParameterBranches(TTree& tree, ShowerFeatures& output, bool includePrimaryEnergy);

// The enriched application clones an existing tree, then fills only these new
// branches. No collision policy is applied here: the historical duplicate-name
// behavior is deliberately characterized until a separate correction is agreed.
class EnrichmentBranches {
public:
    void fill() const;

private:
    friend EnrichmentBranches createEnrichmentBranches(TTree&, ShowerFeatures&);
    std::vector<TBranch*> branches_;
};

EnrichmentBranches createEnrichmentBranches(TTree& tree, ShowerFeatures& output);

}  // namespace sdhcal

#endif  // SDHCAL_ROOT_SHOWER_IO_H
