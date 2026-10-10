/*
  fit.C
  Semi-digital energy reconstruction of pi-, K0_L and p from the threshold
  multiplicities N1, N2, N3, by a weighted least-squares fit with TMinuit.

  Model
  -----
      Nhit    = N1 + N2 + N3
      alpha(Nhit) = a0 + a1*Nhit + a2*Nhit^2
      beta (Nhit) = b0 + b1*Nhit + b2*Nhit^2
      gamma(Nhit) = c0 + c1*Nhit + c2*Nhit^2
      Ereco       = alpha*N1 + beta*N2 + gamma*N3

  Minimised quantity
  ------------------
      S = sum_i (Ereco_i - Etrue_i)^2 / Etrue_i

  What changed with respect to the first version
  ----------------------------------------------
  * THE FIT AND THE FIGURES USED THE SAME EVENTS. The nine coefficients were
    adjusted on every event and the linearity and resolution curves were then
    drawn on those same events, so the method was quoted in-sample while the
    machine-learning models were quoted on a held-out test set: the comparison
    favoured this one. The sample is now split 80/20 with a fixed seed; the fit
    sees only the training part and every figure and number comes from the test
    part.
  * "chi2" WAS A MISNOMER. The denominator is Etrue, not a variance, so the
    quantity above is a weighted least squares and the "chi2/ndf" that used to be
    printed was not a goodness-of-fit test. It is now reported as S/ndf with that
    caveat stated, and the per-point scatter is reported separately.
  * c0 WAS FROZEN BY AN ARBITRARY BOUND. It was declared with limits [0, 1e-10],
    i.e. pinned at a value a thousand times smaller than the other scales, with
    the comment "comme dans ton code". All nine parameters are now free.
  * THE MINUIT STEPS DID NOT MATCH THE PARAMETER SCALES. b2 and c2 are of order
    1e-8 and 1e-7 but were given a step of 1e-5, a hundred to a thousand times
    their own value, which makes the numerical gradient meaningless. Each step is
    now a fixed fraction of the starting value, with a floor.
  * CONVERGENCE WAS NEVER CHECKED. Neither the Migrad return code nor the
    covariance-matrix status was looked at, although the starting values were the
    output of a previous fit. Both are checked and HESSE is run.
  * THE TWO TREES WERE PAIRED BY ROW NUMBER with only a warning when their entry
    counts differed, so a mismatch silently attached each shower to another
    event's energy. The pairing is now by eventNumber when both trees carry it,
    and a count mismatch without that branch is a fatal error.
  * THE RESOLUTION FIT WAS BIASED AT HIGH ENERGY. Every energy bin filled the
    same fixed +-20 GeV histogram, so the Gaussian core was truncated above
    ~40 GeV. The window now scales with the energy, the fit status is checked,
    and the bias is reported next to the resolution instead of being absorbed
    into it.
  * EVENTS ABOVE 100 GeV WERE DROPPED FROM THE FIGURES while the fit used them,
    and the histograms stopped at 100 GeV although the beam goes to 130.
  * THE COEFFICIENTS WERE RETYPED BY HAND into the Python coupling scripts, where
    they had already drifted (kaon c0: 1.22381e-14 here against 9.83127e-15
    there). The fit now writes calibration/fit_<particle>.json; merge the
    fragments into the single calibration file with
        python tools/update_calibration.py

  Input
  -----
      $SDHCAL_DATA_DIR/digitized/130k_<particle>_E1to130_digitized.root  (TTree "tree")
          branches: thr (vector<int>), K (vector<int>), [eventNumber]
      $SDHCAL_DATA_DIR/raw/130k_<particle>_E1to130.root                  (TTree "tree")
          branches: primaryEnergy (double), [eventNumber]

  Output
  ------
      $SDHCAL_OUT_DIR/energy_reco_chi2/plots/ (png figures)
      calibration/fit_<particle>.json       (merge with tools/update_calibration.py)

  Usage
  -----
      root -l -b -q 'fit.C("pi")'
      root -l -b -q 'fit.C("kaon", 0.2, 42, true)'
      root -l -b -q 'fit.C+("proton")'      // compiled, recommended
*/

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TF1.h"
#include "TFile.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TH1D.h"
#include "TLegend.h"
#include "TLine.h"
#include "TMath.h"
#include "TMinuit.h"
#include "TProfile.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TTree.h"

#include "../../../cpp/include/SdhcalPaths.h"

// ---------------------------------------------------------------------------
// Data shared with the minimisation function. Only the TRAINING events are put
// here, so the objective cannot see the test sample.
// ---------------------------------------------------------------------------
namespace {

std::vector<int> gN1, gN2, gN3;
std::vector<double> gEbeam;

constexpr int kNPar = 9;
const char* kParNames[kNPar] = {"a0", "a1", "a2", "b0", "b1", "b2", "c0", "c1", "c2"};

struct Event {
  int n1 = 0, n2 = 0, n3 = 0;
  double eBeam = 0.0;
};

/// Weighted least squares: sum (Ereco - Etrue)^2 / Etrue.
/// Not a chi2: the denominator is an energy, not a variance.
void wlsFunction(int& /*npar*/, double* /*grad*/, double& fval, double* par, int /*flag*/) {
  double sum = 0.0;
  for (std::size_t i = 0; i < gN1.size(); ++i) {
    const double nhit = static_cast<double>(gN1[i] + gN2[i] + gN3[i]);
    const double alpha = par[0] + par[1] * nhit + par[2] * nhit * nhit;
    const double beta = par[3] + par[4] * nhit + par[5] * nhit * nhit;
    const double gamma = par[6] + par[7] * nhit + par[8] * nhit * nhit;
    const double eReco = alpha * gN1[i] + beta * gN2[i] + gamma * gN3[i];
    const double diff = eReco - gEbeam[i];
    sum += diff * diff / gEbeam[i];
  }
  fval = std::isfinite(sum) ? sum : 1e30;
}

double energyOf(const Event& event, const double* par) {
  const double nhit = static_cast<double>(event.n1 + event.n2 + event.n3);
  const double alpha = par[0] + par[1] * nhit + par[2] * nhit * nhit;
  const double beta = par[3] + par[4] * nhit + par[5] * nhit * nhit;
  const double gamma = par[6] + par[7] * nhit + par[8] * nhit * nhit;
  return alpha * event.n1 + beta * event.n2 + gamma * event.n3;
}

bool hasBranch(TTree* tree, const char* name) {
  return tree && tree->GetListOfBranches()->FindObject(name) != nullptr;
}

/// Starting values: a scale-free guess, so the fit no longer starts from its own
/// previous answer. alpha ~ E/Nhit at the few-10 GeV scale, beta and gamma above
/// it because higher thresholds flag denser regions.
void startingValues(double* start) {
  start[0] = 4.0e-2;  start[1] = -2.0e-5;  start[2] = 1.0e-8;   // alpha
  start[3] = 1.0e-1;  start[4] = -5.0e-5;  start[5] = 1.0e-8;   // beta
  start[6] = 1.0e-1;  start[7] = 5.0e-4;   start[8] = -2.0e-7;  // gamma
}

}  // namespace

// ---------------------------------------------------------------------------
void fit(const std::string& particle,
                double testFraction = 0.2,
                unsigned int seed = 42,
                bool requireContained = true) {
  gN1.clear(); gN2.clear(); gN3.clear(); gEbeam.clear();

  if (particle != "pi" && particle != "kaon" && particle != "proton") {
    std::cerr << "Unknown particle '" << particle << "'; expected pi, kaon or proton\n";
    return;
  }
  if (testFraction <= 0.0 || testFraction >= 1.0) {
    std::cerr << "testFraction must lie strictly between 0 and 1\n";
    return;
  }

  // --- Input files -------------------------------------------------------
  const std::string hitsName = sdhcal::digitized(particle);
  const std::string energyName = sdhcal::raw(particle);

  TFile* fHits = TFile::Open(hitsName.c_str());
  TFile* fEnergy = TFile::Open(energyName.c_str());
  if (!fHits || fHits->IsZombie() || !fEnergy || fEnergy->IsZombie()) {
    std::cerr << "Cannot open the inputs for " << particle << ":\n  " << hitsName << "\n  "
              << energyName << "\nSet SDHCAL_DATA_DIR to the directory holding the ROOT files.\n";
    return;
  }

  TTree* treeHits = dynamic_cast<TTree*>(fHits->Get("tree"));
  TTree* treeEnergy = dynamic_cast<TTree*>(fEnergy->Get("tree"));
  if (!treeHits || !treeEnergy) {
    std::cerr << "TTree 'tree' missing in one of the inputs.\n";
    return;
  }

  const Long64_t nHits = treeHits->GetEntries();
  const Long64_t nEnergy = treeEnergy->GetEntries();
  const bool pairById = hasBranch(treeHits, "eventNumber") && hasBranch(treeEnergy, "eventNumber");

  if (!pairById) {
    if (nHits != nEnergy) {
      std::cerr << "FATAL: the two trees hold different numbers of entries (hits=" << nHits
                << ", energy=" << nEnergy << ") and neither carries an 'eventNumber' branch, so "
                   "the showers cannot be matched to their energies. Re-produce the files with "
                   "an event identifier.\n";
      return;
    }
    std::cerr << "WARNING: no 'eventNumber' branch; falling back to pairing by row number. "
                 "This is only correct if the two files were written in lockstep.\n";
  }

  // --- Read the energies -------------------------------------------------
  double primaryEnergy = 0.0;
  Int_t energyEventId = 0;
  treeEnergy->SetBranchAddress("primaryEnergy", &primaryEnergy);
  if (pairById) treeEnergy->SetBranchAddress("eventNumber", &energyEventId);

  std::map<Int_t, double> energyById;
  std::vector<double> energyByRow;
  energyByRow.reserve(static_cast<std::size_t>(nEnergy));
  for (Long64_t i = 0; i < nEnergy; ++i) {
    treeEnergy->GetEntry(i);
    if (pairById) energyById[energyEventId] = primaryEnergy;
    else energyByRow.push_back(primaryEnergy);
  }

  // --- Read the hits and build the event list ----------------------------
  std::vector<int>* thr = nullptr;
  std::vector<int>* layer = nullptr;
  Int_t hitsEventId = 0;
  treeHits->SetBranchAddress("thr", &thr);
  treeHits->SetBranchAddress("K", &layer);
  if (pairById) treeHits->SetBranchAddress("eventNumber", &hitsEventId);

  const int kLastLayer = 47;
  std::vector<Event> events;
  events.reserve(static_cast<std::size_t>(nHits));
  Long64_t nNotContained = 0, nUnmatched = 0, nBadEnergy = 0;

  for (Long64_t i = 0; i < nHits; ++i) {
    treeHits->GetEntry(i);

    double eBeam = 0.0;
    if (pairById) {
      const auto found = energyById.find(hitsEventId);
      if (found == energyById.end()) { ++nUnmatched; continue; }
      eBeam = found->second;
    } else {
      eBeam = energyByRow[static_cast<std::size_t>(i)];
    }
    if (!(eBeam > 0.0)) { ++nBadEnergy; continue; }

    if (requireContained) {
      bool touchesLast = false;
      for (std::size_t h = 0; h < layer->size(); ++h) {
        if ((*layer)[h] == kLastLayer) { touchesLast = true; break; }
      }
      if (touchesLast) { ++nNotContained; continue; }
    }

    Event event;
    event.eBeam = eBeam;
    for (std::size_t h = 0; h < thr->size(); ++h) {
      switch ((*thr)[h]) {
        case 1: ++event.n1; break;
        case 2: ++event.n2; break;
        case 3: ++event.n3; break;
        default: break;
      }
    }
    if (event.n1 + event.n2 + event.n3 == 0) continue;
    events.push_back(event);
  }

  if (events.size() < 100) {
    std::cerr << "Only " << events.size() << " usable events; aborting.\n";
    return;
  }

  std::cout << "=== " << particle << " ===\n"
            << "  usable events        : " << events.size() << "\n"
            << "  rejected, last layer : " << nNotContained
            << (requireContained ? "  (containment cut ON)\n" : "  (cut OFF)\n")
            << "  rejected, no energy  : " << nBadEnergy << "\n"
            << "  unmatched events     : " << nUnmatched << "\n";

  // --- Train / test split, fixed seed ------------------------------------
  std::vector<std::size_t> order(events.size());
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), std::mt19937_64(seed));

  const std::size_t nTest = static_cast<std::size_t>(testFraction * events.size());
  const std::vector<std::size_t> testIdx(order.begin(), order.begin() + nTest);
  const std::vector<std::size_t> trainIdx(order.begin() + nTest, order.end());

  gN1.reserve(trainIdx.size()); gN2.reserve(trainIdx.size());
  gN3.reserve(trainIdx.size()); gEbeam.reserve(trainIdx.size());
  for (std::size_t k : trainIdx) {
    gN1.push_back(events[k].n1);
    gN2.push_back(events[k].n2);
    gN3.push_back(events[k].n3);
    gEbeam.push_back(events[k].eBeam);
  }
  std::cout << "  train / test         : " << trainIdx.size() << " / " << testIdx.size()
            << "   (seed " << seed << ")\n";

  // --- Minimisation ------------------------------------------------------
  double start[kNPar];
  startingValues(start);

  TMinuit minuit(kNPar);
  minuit.SetFCN(wlsFunction);
  double arglist[2];
  Int_t ierr = 0;
  arglist[0] = -1;  // quiet
  minuit.mnexcm("SET PRINT", arglist, 1, ierr);

  for (int ip = 0; ip < kNPar; ++ip) {
    // The step must follow the scale of the parameter it moves, otherwise the
    // numerical derivative is dominated by rounding.
    const double step = std::max(1e-3 * std::fabs(start[ip]), 1e-12);
    // No limits: every coefficient is free.
    minuit.DefineParameter(ip, kParNames[ip], start[ip], step, 0.0, 0.0);
  }

  arglist[0] = 50000;  // max calls
  arglist[1] = 1e-6;   // tolerance
  minuit.mnexcm("MIGRAD", arglist, 2, ierr);
  const Int_t migradStatus = ierr;
  minuit.mnexcm("HESSE", arglist, 1, ierr);

  double fmin = 0.0, fedm = 0.0, errdef = 0.0;
  Int_t nvpar = 0, nparx = 0, istat = 0;
  minuit.mnstat(fmin, fedm, errdef, nvpar, nparx, istat);

  double par[kNPar], err[kNPar];
  for (int ip = 0; ip < kNPar; ++ip) minuit.GetParameter(ip, par[ip], err[ip]);

  const int ndf = static_cast<int>(trainIdx.size()) - nvpar;
  std::cout << "  --- minimisation ---\n"
            << "  MIGRAD status        : " << migradStatus << (migradStatus == 0 ? "  (converged)\n" : "  (DID NOT CONVERGE)\n")
            << "  covariance status    : " << istat
            << "  (3 = accurate, <3 = approximate or not positive-definite)\n"
            << "  S_min                : " << fmin << "   ndf = " << ndf
            << "   S/ndf = " << (ndf > 0 ? fmin / ndf : -1.0) << "\n"
            << "  NOTE: S/ndf is NOT a goodness-of-fit chi2/ndf; the weight is 1/E, not 1/sigma^2.\n"
            << "  estimated distance to minimum: " << fedm << "\n";
  if (migradStatus != 0 || istat < 3) {
    std::cerr << "  WARNING: the fit did not fully converge; the coefficients below are not reliable.\n";
  }
  for (int ip = 0; ip < kNPar; ++ip) {
    std::cout << "    " << kParNames[ip] << " = " << par[ip] << " +- " << err[ip] << "\n";
  }

  // --- Evaluation, on the TEST sample only -------------------------------
  std::vector<double> eTrue, eReco;
  eTrue.reserve(testIdx.size());
  eReco.reserve(testIdx.size());
  double eMax = 0.0;
  for (std::size_t k : testIdx) {
    eTrue.push_back(events[k].eBeam);
    eReco.push_back(energyOf(events[k], par));
    eMax = std::max(eMax, events[k].eBeam);
  }
  const double eUpper = std::ceil(eMax / 10.0) * 10.0;  // 130 GeV, not a hard-coded 100

  const std::string plotDir = sdhcal::outputDir() + "/energy_reco_chi2/plots";
  gSystem->mkdir(plotDir.c_str(), kTRUE);
  const std::string tag = "_" + particle;

  TH1D* hReco = new TH1D(("hEreco" + tag).c_str(), "E_{reco};E [GeV];events", 130, 0, eUpper);
  TH1D* hBeam = new TH1D(("hEbeam" + tag).c_str(), "E_{beam};E [GeV];events", 130, 0, eUpper);
  TProfile* profile = new TProfile(("pProf" + tag).c_str(),
                                   "Linearity (test sample);E_{true} [GeV];#LTE_{reco}#GT [GeV]",
                                   65, 0, eUpper);
  hReco->SetDirectory(nullptr);
  hBeam->SetDirectory(nullptr);
  for (std::size_t i = 0; i < eTrue.size(); ++i) {
    hReco->Fill(eReco[i]);
    hBeam->Fill(eTrue[i]);
    profile->Fill(eTrue[i], eReco[i]);
  }

  TCanvas* cComp = new TCanvas(("cComp" + tag).c_str(), "Reco vs beam", 800, 600);
  hBeam->SetStats(kFALSE);
  hBeam->SetLineColor(kBlue);
  hReco->SetLineColor(kRed);
  hBeam->Draw("HIST");
  hReco->Draw("HIST SAME");
  TLegend* legend = new TLegend(0.60, 0.75, 0.88, 0.88);
  legend->AddEntry(hBeam, "E beam", "l");
  legend->AddEntry(hReco, "E reco", "l");
  legend->Draw();
  cComp->SaveAs((plotDir + "/Reco_vs_Beam" + tag + ".png").c_str());

  TCanvas* cProf = new TCanvas(("cProf" + tag).c_str(), "Linearity", 800, 600);
  profile->SetStats(kFALSE);
  profile->GetYaxis()->SetRangeUser(0, eUpper);
  profile->Draw();
  TLine* diagonal = new TLine(0, 0, eUpper, eUpper);
  diagonal->SetLineColor(kRed);
  diagonal->SetLineStyle(2);
  diagonal->Draw();
  cProf->SaveAs((plotDir + "/Profil_linearite" + tag + ".png").c_str());

  TCanvas* cScatter = new TCanvas(("cScatter" + tag).c_str(), "E reco vs E true", 800, 600);
  TGraph* scatter = new TGraph(static_cast<int>(eTrue.size()), eTrue.data(), eReco.data());
  scatter->SetTitle("E_{reco} vs E_{beam} (test sample);E_{beam} [GeV];E_{reco} [GeV]");
  scatter->SetMarkerStyle(1);
  scatter->GetYaxis()->SetRangeUser(0, eUpper);
  scatter->Draw("AP");
  TLine* diagonal2 = new TLine(0, 0, eUpper, eUpper);
  diagonal2->SetLineColor(kRed);
  diagonal2->SetLineStyle(2);
  diagonal2->SetLineWidth(2);
  diagonal2->Draw();
  cScatter->SaveAs((plotDir + "/Linearity" + tag + ".png").c_str());

  // --- Bias and resolution, separately -----------------------------------
  const int nBins = 13;
  const double eLow = 0.0;
  const double dE = (eUpper - eLow) / nBins;

  std::vector<double> center, bias, biasErr, sigma, sigmaErr;
  for (int ib = 0; ib < nBins; ++ib) {
    const double lo = eLow + ib * dE;
    const double hi = lo + dE;
    const double mid = 0.5 * (lo + hi);
    if (mid <= 0.0) continue;

    // The window follows the energy instead of being frozen at +-20 GeV, which
    // used to cut into the Gaussian core above ~40 GeV.
    TH1D hres(("hres" + tag + std::to_string(ib)).c_str(),
              "#DeltaE/E;(E_{reco}-E_{true})/E_{true};events", 80, -1.5, 1.5);
    for (std::size_t i = 0; i < eTrue.size(); ++i) {
      if (eTrue[i] >= lo && eTrue[i] < hi) hres.Fill((eReco[i] - eTrue[i]) / eTrue[i]);
    }
    if (hres.GetEntries() < 50) continue;

    // Fit the core within +-2 RMS of the mean, so the tails do not drive sigma.
    const double guessMean = hres.GetMean();
    const double guessRms = hres.GetRMS();
    // "Q" quiet, "0" do not draw. Not "N": that would skip storing the function,
    // and GetFunction("gaus") below would then return a null pointer.
    const int status = hres.Fit("gaus", "Q0", "", guessMean - 2.0 * guessRms, guessMean + 2.0 * guessRms);
    TF1* fit = hres.GetFunction("gaus");
    if (status != 0 || !fit) {
      std::cerr << "  bin " << ib << " [" << lo << ", " << hi << "] GeV: Gaussian fit failed, "
                   "falling back to histogram mean and RMS\n";
      center.push_back(mid);
      bias.push_back(guessMean);
      biasErr.push_back(hres.GetMeanError());
      sigma.push_back(guessRms);
      sigmaErr.push_back(hres.GetRMSError());
      continue;
    }
    center.push_back(mid);
    bias.push_back(fit->GetParameter(1));
    biasErr.push_back(fit->GetParError(1));
    sigma.push_back(std::fabs(fit->GetParameter(2)));
    sigmaErr.push_back(fit->GetParError(2));
  }

  if (!center.empty()) {
    TCanvas* cRes = new TCanvas(("cRes" + tag).c_str(), "Bias and resolution", 900, 700);
    cRes->Divide(1, 2);

    cRes->cd(1);
    TGraphErrors* gBias = new TGraphErrors(static_cast<int>(center.size()), center.data(), bias.data(),
                                           nullptr, biasErr.data());
    gBias->SetTitle("Bias (test sample);E_{true} [GeV];#LT#DeltaE/E#GT");
    gBias->SetMarkerStyle(20);
    gBias->Draw("AP");
    TLine* zero = new TLine(0, 0, eUpper, 0);
    zero->SetLineStyle(2);
    zero->Draw();

    cRes->cd(2);
    TGraphErrors* gRes = new TGraphErrors(static_cast<int>(center.size()), center.data(), sigma.data(),
                                          nullptr, sigmaErr.data());
    gRes->SetTitle("Relative resolution (test sample);E_{true} [GeV];#sigma(#DeltaE/E)");
    gRes->SetMarkerStyle(20);
    gRes->Draw("AP");

    cRes->SaveAs((plotDir + "/Bias_and_Resolution" + tag + ".png").c_str());
  }

  // --- Global test numbers ----------------------------------------------
  double sumRel = 0.0, sumRel2 = 0.0, sumTrue = 0.0, sumReco = 0.0;
  for (std::size_t i = 0; i < eTrue.size(); ++i) {
    const double r = (eReco[i] - eTrue[i]) / eTrue[i];
    sumRel += r;
    sumRel2 += r * r;
    sumTrue += eTrue[i];
    sumReco += eReco[i];
  }
  const double n = static_cast<double>(eTrue.size());
  const double meanRel = sumRel / n;
  const double rmsRel = std::sqrt(sumRel2 / n);
  std::cout << "  --- test sample (" << eTrue.size() << " events) ---\n"
            << "  mean relative residual : " << meanRel << "\n"
            << "  RMS about zero         : " << rmsRel
            << "   (= sqrt(bias^2 + sigma^2), not a resolution)\n"
            << "  energy scale <Ereco>/<Etrue> : " << (sumReco / sumTrue) << "\n";

  // --- Coefficients, for the shared calibration file ---------------------
  gSystem->mkdir(sdhcal::calibrationDir().c_str(), kTRUE);
  const std::string fragment = sdhcal::calibrationDir() + "/fit_" + particle + ".json";
  std::ofstream out(fragment);
  if (out) {
    out.precision(10);
    out << std::scientific;
    out << "{\n  \"particle\": \"" << particle << "\",\n";
    for (int ip = 0; ip < kNPar; ++ip) out << "  \"" << kParNames[ip] << "\": " << par[ip] << ",\n";
    out << "  \"errors\": {\n";
    for (int ip = 0; ip < kNPar; ++ip) {
      out << "    \"" << kParNames[ip] << "\": " << err[ip] << (ip + 1 < kNPar ? ",\n" : "\n");
    }
    out << "  },\n";
    out << "  \"source\": \"fit.C weighted least squares\",\n";
    out << "  \"fitted_on_train_split\": true,\n";
    out << "  \"n_train\": " << trainIdx.size() << ",\n";
    out << "  \"n_test\": " << testIdx.size() << ",\n";
    out << "  \"test_fraction\": " << testFraction << ",\n";
    out << "  \"split_seed\": " << seed << ",\n";
    out << "  \"migrad_status\": " << migradStatus << ",\n";
    out << "  \"covariance_status\": " << istat << ",\n";
    out << "  \"objective\": \"sum (Ereco-Etrue)^2 / Etrue\",\n";
    out << "  \"selection\": \"" << (requireContained ? "no hit in layer 47, " : "")
        << "primaryEnergy > 0\"\n";
    out << "}\n";
    out.close();
    std::cout << "  coefficients written to " << fragment << "\n"
              << "  merge them with: python tools/update_calibration.py\n";
  } else {
    std::cerr << "  could not write " << fragment << "\n";
  }

  fHits->Close();
  fEnergy->Close();
}
