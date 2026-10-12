// Publication-style comparison plots for the model benchmark.
//
//   root -l -b -q 'analysis/6_benchmark/plots.C("energy")'
//   root -l -b -q 'analysis/6_benchmark/plots.C("pid")'
//
// Reads the tables written by tools/compare_models.py and draws them. The numbers
// are computed in Python, where they are tested; this file only draws, so a plot
// can be restyled without recomputing anything and a number cannot change by
// being plotted differently.
//
// Input, under $SDHCAL_OUT_DIR/benchmark/<kind>/tables/ :
//   summary.csv        one row per model
//   curve_<model>.csv  energy: energy,response,bias,bias_err,sigma,sigma_err,...
//   roc_<model>.csv    pid:    efficiency_signal,efficiency_background,threshold
//   eff_<model>.csv    pid:    energy,efficiency_signal,efficiency_signal_err,...
//
// Output: PNG and PDF next to them, in ../plots/.

#include "TAxis.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TLine.h"
#include "TList.h"
#include "TMultiGraph.h"
#include "TF1.h"
#include "TStyle.h"
#include "TSystem.h"

#include <clocale>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

const int kColour[] = {kAzure + 2, kRed + 1, kGreen + 2, kOrange + 7, kMagenta + 2, kCyan + 2};
const int kMarker[] = {20, 21, 22, 23, 33, 34};

std::string outputDir() {
  const char* env = std::getenv("SDHCAL_OUT_DIR");
  return env && *env ? std::string(env) : std::string("output");
}

/// Minimal CSV reader: header line, then numbers. Enough for our own tables, and
/// deliberately strict - a missing column is a mistake upstream, not something to
/// paper over here.
struct Table {
  std::map<std::string, std::vector<double> > columns;
  std::vector<std::string> order;
  std::vector<std::string> text;  // first column when it is not numeric

  bool has(const std::string& name) const { return columns.count(name) > 0; }
  const std::vector<double>& operator[](const std::string& name) const {
    auto it = columns.find(name);
    if (it == columns.end()) {
      std::cerr << "missing column '" << name << "'\n";
      std::exit(1);
    }
    return it->second;
  }
  size_t size() const { return order.empty() ? 0 : columns.at(order.front()).size(); }
};

Table readCsv(const std::string& path) {
  // The tables are written by Python with a dot decimal separator, but strtod
  // follows the process locale, and a fr_FR locale parses "0.33" as 0. That cost
  // a plot that silently lost the middle of every curve, so the numeric locale is
  // pinned here rather than trusted.
  std::setlocale(LC_NUMERIC, "C");

  Table table;
  std::ifstream in(path.c_str());
  if (!in) {
    std::cerr << "cannot open " << path << "\n";
    std::exit(1);
  }
  std::string line;
  if (!std::getline(in, line)) return table;

  std::stringstream header(line);
  std::string name;
  while (std::getline(header, name, ',')) {
    table.order.push_back(name);
    table.columns[name] = std::vector<double>();
  }

  while (std::getline(in, line)) {
    std::stringstream row(line);
    std::string cell;
    for (size_t i = 0; i < table.order.size() && std::getline(row, cell, ','); ++i) {
      if (i == 0 && !cell.empty() && (std::isalpha(cell[0]) || cell[0] == '_')) {
        table.text.push_back(cell);
        table.columns[table.order[i]].push_back(0.0);
        continue;
      }
      table.columns[table.order[i]].push_back(cell.empty() ? 0.0 : std::strtod(cell.c_str(), nullptr));
    }
  }
  return table;
}

void applyStyle() {
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetPadTickX(1);          // ticks on all four sides, as HEP plots have
  gStyle->SetPadTickY(1);
  gStyle->SetPadLeftMargin(0.14);
  gStyle->SetPadBottomMargin(0.13);
  gStyle->SetPadRightMargin(0.04);
  gStyle->SetPadTopMargin(0.06);
  gStyle->SetTitleSize(0.048, "XYZ");
  gStyle->SetLabelSize(0.042, "XYZ");
  gStyle->SetTitleOffset(1.30, "Y");
  gStyle->SetTitleOffset(1.10, "X");
  gStyle->SetLegendBorderSize(0);
  // Transparent, not white: an opaque legend box painted over the rejection
  // curve once looked exactly like a gap in the data.
  gStyle->SetLegendFillColor(0);
  gStyle->SetLegendFillStyle(0);
  gStyle->SetLegendFont(42);
  gStyle->SetTextFont(42);
  gStyle->SetEndErrorSize(3);
}

/// The identification block every HEP plot carries: who, what sample, what cuts.
void drawHeader(const char* detail) {
  TLatex latex;
  latex.SetNDC();
  latex.SetTextFont(62);
  latex.SetTextSize(0.050);
  latex.DrawLatex(0.14, 0.955, "SDHCAL");
  latex.SetTextFont(52);
  latex.SetTextSize(0.040);
  latex.DrawLatex(0.285, 0.955, "Simulation");
  latex.SetTextFont(42);
  latex.SetTextSize(0.037);
  latex.SetTextAlign(31);
  latex.DrawLatex(0.96, 0.955, detail);
}

void save(TCanvas& canvas, const std::string& stem) {
  canvas.SaveAs((stem + ".png").c_str());
  canvas.SaveAs((stem + ".pdf").c_str());
}

std::vector<std::string> modelNames(const std::string& dir) {
  Table summary = readCsv(dir + "/summary.csv");
  return summary.text;
}

// ---------------------------------------------------------------------------
// Energy
// ---------------------------------------------------------------------------
void plotEnergy(const std::string& dir, const std::string& outDir) {
  const std::vector<std::string> models = modelNames(dir);
  Table summary = readCsv(dir + "/summary.csv");

  TCanvas cRes("cRes", "", 800, 650);
  cRes.SetLogx();
  TMultiGraph mgRes;
  TLegend legRes(0.46, 0.63, 0.95, 0.90);
  legRes.SetTextSize(0.032);

  TCanvas cLin("cLin", "", 800, 650);
  cLin.SetLogx();
  TMultiGraph mgLin;
  TLegend legLin(0.46, 0.70, 0.95, 0.90);
  legLin.SetTextSize(0.032);

  for (size_t i = 0; i < models.size(); ++i) {
    Table curve = readCsv(dir + "/curve_" + models[i] + ".csv");
    const size_t n = curve.size();
    if (n == 0) continue;

    TGraphErrors* gr = new TGraphErrors(static_cast<int>(n));
    TGraphErrors* gl = new TGraphErrors(static_cast<int>(n));
    for (size_t k = 0; k < n; ++k) {
      gr->SetPoint(k, curve["energy"][k], curve["sigma"][k]);
      gr->SetPointError(k, 0.0, curve["sigma_err"][k]);
      gl->SetPoint(k, curve["energy"][k], curve["response"][k]);
      gl->SetPointError(k, 0.0, curve["bias_err"][k]);
    }
    for (TGraphErrors* g : {gr, gl}) {
      g->SetMarkerStyle(kMarker[i % 6]);
      g->SetMarkerColor(kColour[i % 6]);
      g->SetLineColor(kColour[i % 6]);
      g->SetMarkerSize(1.15);
    }

    // The fitted two-term form, drawn over the points it was fitted to.
    const double a = summary["stochastic_term"][i];
    const double b = summary["constant_term"][i];
    TF1* f = new TF1(("f_" + models[i]).c_str(), "sqrt([0]*[0]/x + [1]*[1])",
                     curve["energy"][0], curve["energy"][n - 1]);
    f->SetParameters(a, b);
    f->SetLineColor(kColour[i % 6]);
    f->SetLineStyle(2);
    f->SetLineWidth(2);

    mgRes.Add(gr, "P");
    mgLin.Add(gl, "P");
    legRes.AddEntry(gr,
                    Form("%s:  a = %.1f%%,  b = %.1f%%", models[i].c_str(), 100 * a, 100 * b), "lp");
    legLin.AddEntry(gl, models[i].c_str(), "lp");
    cRes.cd();
    f->Draw("SAME");
  }

  cRes.cd();
  mgRes.Draw("A");
  mgRes.GetXaxis()->SetTitle("E_{true} [GeV]");
  mgRes.GetYaxis()->SetTitle("#sigma(E)/E");
  mgRes.GetYaxis()->SetRangeUser(0.0, 0.45);
  // The multigraph pads a log axis down to a decade the sample does not cover.
  mgRes.GetXaxis()->SetLimits(4.0, 130.0);
  // Redraw the fitted curves on top of the axes created by the multigraph.
  for (size_t i = 0; i < models.size(); ++i) {
    TF1* f = (TF1*)gROOT->FindObject(("f_" + models[i]).c_str());
    if (f) f->Draw("SAME");
  }
  legRes.SetHeader("#sigma/E = a/#sqrt{E} #oplus b");
  legRes.Draw();
  drawHeader("#pi^{-}/p, fixed-energy sample");
  save(cRes, outDir + "/energy_resolution");

  cLin.cd();
  mgLin.Draw("AP");
  mgLin.GetXaxis()->SetTitle("E_{true} [GeV]");
  mgLin.GetYaxis()->SetTitle("#LT E_{rec} #GT / E_{true}");
  mgLin.GetYaxis()->SetRangeUser(0.90, 1.12);
  mgLin.GetXaxis()->SetLimits(4.0, 130.0);
  TF1 unity("unity", "1.0", 4.0, 130.0);
  unity.SetLineColor(kGray + 2);
  unity.SetLineStyle(3);
  unity.Draw("SAME");
  legLin.SetHeader("Response");
  legLin.Draw();
  drawHeader("#pi^{-}/p, fixed-energy sample");
  save(cLin, outDir + "/energy_linearity");
}

// ---------------------------------------------------------------------------
// Identification
// ---------------------------------------------------------------------------
void plotPid(const std::string& dir, const std::string& outDir) {
  const std::vector<std::string> models = modelNames(dir);
  Table summary = readCsv(dir + "/summary.csv");

  TCanvas cRoc("cRoc", "", 800, 650);
  cRoc.SetLogy();
  TMultiGraph mgRoc;
  TLegend legRoc(0.50, 0.74, 0.95, 0.90);
  legRoc.SetTextSize(0.032);

  TCanvas cEff("cEff", "", 800, 650);
  cEff.SetLogx();
  TMultiGraph mgEff;
  TLegend legEff(0.46, 0.73, 0.95, 0.90);
  legEff.SetTextSize(0.032);

  for (size_t i = 0; i < models.size(); ++i) {
    Table roc = readCsv(dir + "/roc_" + models[i] + ".csv");
    TGraph* g = new TGraph();
    int point = 0;
    for (size_t k = 0; k < roc.size(); ++k) {
      const double effB = roc["efficiency_background"][k];
      const double effS = roc["efficiency_signal"][k];
      if (effB <= 0.0 || effS <= 0.0) continue;   // rejection is 1/effB
      g->SetPoint(point++, effS, 1.0 / effB);
    }
    g->SetLineColor(kColour[i % 6]);
    g->SetLineWidth(2);
    mgRoc.Add(g, "L");
    legRoc.AddEntry(g, Form("%s:  AUC = %.3f", models[i].c_str(), summary["auc"][i]), "l");

    Table eff = readCsv(dir + "/eff_" + models[i] + ".csv");
    TGraphErrors* ge = new TGraphErrors(static_cast<int>(eff.size()));
    for (size_t k = 0; k < eff.size(); ++k) {
      ge->SetPoint(k, eff["energy"][k], eff["efficiency_signal"][k]);
      ge->SetPointError(k, 0.0, eff["efficiency_signal_err"][k]);
    }
    ge->SetMarkerStyle(kMarker[i % 6]);
    ge->SetMarkerColor(kColour[i % 6]);
    ge->SetLineColor(kColour[i % 6]);
    ge->SetMarkerSize(1.15);
    mgEff.Add(ge, "P");
    legEff.AddEntry(ge, models[i].c_str(), "lp");
  }

  cRoc.cd();
  mgRoc.Draw("A");
  mgRoc.GetXaxis()->SetTitle("#pi^{-} efficiency");
  mgRoc.GetYaxis()->SetTitle("proton rejection  1/#varepsilon_{p}");
  mgRoc.GetXaxis()->SetLimits(0.0, 1.0);
  legRoc.Draw();
  drawHeader("fixed-energy sample");
  save(cRoc, outDir + "/pid_rejection");

  cEff.cd();
  mgEff.Draw("A");
  mgEff.GetXaxis()->SetTitle("E_{true} [GeV]");
  mgEff.GetYaxis()->SetTitle("#pi^{-} efficiency");
  mgEff.GetYaxis()->SetRangeUser(0.0, 1.0);
  mgEff.GetXaxis()->SetLimits(4.0, 130.0);
  legEff.SetHeader("at the 50% working point");
  legEff.Draw();
  drawHeader("fixed-energy sample");
  save(cEff, outDir + "/pid_efficiency_vs_energy");
}


// ---------------------------------------------------------------------------
// PID-energy coupling
// ---------------------------------------------------------------------------
// One figure per regressor, so that the comparison the text makes - what the
// species label is worth, at fixed regressor - is the comparison the eye makes.
// The three label sources sit on one pad; the two regressors never share one.
struct Scenario {
  const char* file;    // stem of the curve table
  const char* legend;
};

void plotCouplingArm(const std::string& dir, const std::string& outDir, const char* reco,
                     const char* recoLabel, const std::vector<Scenario>& arms,
                     const char* species, const char* speciesLabel) {
  TMultiGraph mgRes;
  TMultiGraph mgLin;
  TLegend legRes(0.58, 0.72, 0.95, 0.90);
  TLegend legLin(0.58, 0.17, 0.95, 0.35);
  legRes.SetTextSize(0.032);
  legLin.SetTextSize(0.032);

  for (size_t i = 0; i < arms.size(); ++i) {
    const std::string path = dir + "/" + arms[i].file + "_reco-" + reco + "__" + species + ".csv";
    if (gSystem->AccessPathName(path.c_str())) {
      std::cerr << "missing " << path << "\n";
      continue;
    }
    Table t = readCsv(path);
    const size_t n = t.columns.at("energy").size();

    auto* gr = new TGraphErrors(n, t.columns.at("energy").data(), t.columns.at("resolution").data(),
                                nullptr, t.columns.at("resolution_err").data());
    auto* gl = new TGraphErrors(n, t.columns.at("energy").data(), t.columns.at("response").data(),
                                nullptr, t.columns.at("response_err").data());
    for (auto* g : {gr, gl}) {
      g->SetMarkerStyle(kMarker[i % 6]);
      g->SetMarkerColor(kColour[i % 6]);
      g->SetLineColor(kColour[i % 6]);
      g->SetMarkerSize(1.15);
    }
    mgRes.Add(gr, "P");
    mgLin.Add(gl, "P");
    legRes.AddEntry(gr, arms[i].legend, "p");
    legLin.AddEntry(gl, arms[i].legend, "p");
  }

  if (mgRes.GetListOfGraphs() == nullptr || mgRes.GetListOfGraphs()->GetSize() == 0) {
    std::cerr << "no curve for reco=" << reco << " species=" << species << "; figure skipped\n";
    return;
  }

  const std::string tag = std::string(reco) + "_" + species;
  const std::string detail = std::string(speciesLabel) + ", " + recoLabel;

  TCanvas cRes(("cRes_" + tag).c_str(), "", 800, 600);
  cRes.SetLogx();
  mgRes.Draw("A");
  mgRes.GetXaxis()->SetTitle("E_{true} [GeV]");
  mgRes.GetYaxis()->SetTitle("#sigma(E)/E");
  mgRes.GetYaxis()->SetRangeUser(0.0, 0.30);
  mgRes.GetXaxis()->SetLimits(4.0, 130.0);
  legRes.Draw();
  drawHeader(detail.c_str());
  save(cRes, outDir + "/coupling_resolution_" + tag);

  TCanvas cLin(("cLin_" + tag).c_str(), "", 800, 600);
  cLin.SetLogx();
  mgLin.Draw("A");
  mgLin.GetXaxis()->SetTitle("E_{true} [GeV]");
  mgLin.GetYaxis()->SetTitle("#LT E_{rec} #GT / E_{true}");
  mgLin.GetYaxis()->SetRangeUser(0.90, 1.12);
  mgLin.GetXaxis()->SetLimits(4.0, 130.0);
  auto* unity = new TLine(mgLin.GetXaxis()->GetXmin(), 1.0, mgLin.GetXaxis()->GetXmax(), 1.0);
  unity->SetLineStyle(3);
  unity->SetLineColor(kGray + 2);
  unity->Draw();
  legLin.Draw();
  drawHeader(detail.c_str());
  save(cLin, outDir + "/coupling_linearity_" + tag);
}


// One scenario, the two species side by side: what chapter 7 shows, where the
// question is how a single estimator treats pions and protons differently.
void plotCouplingSpecies(const std::string& dir, const std::string& outDir,
                         const char* scenario, const char* detail) {
  const std::vector<std::pair<const char*, const char*>> species = {{"pi-", "#pi^{-}"}, {"p", "p"}};
  TMultiGraph mgRes;
  TMultiGraph mgLin;
  TLegend legRes(0.70, 0.74, 0.95, 0.90);
  TLegend legLin(0.70, 0.17, 0.95, 0.33);
  legRes.SetTextSize(0.034);
  legLin.SetTextSize(0.034);

  for (size_t i = 0; i < species.size(); ++i) {
    const std::string path = dir + "/" + scenario + "__" + species[i].first + ".csv";
    if (gSystem->AccessPathName(path.c_str())) {
      std::cerr << "missing " << path << "\n";
      continue;
    }
    Table t = readCsv(path);
    const size_t n = t.columns.at("energy").size();
    auto* gr = new TGraphErrors(n, t.columns.at("energy").data(), t.columns.at("resolution").data(),
                                nullptr, t.columns.at("resolution_err").data());
    auto* gl = new TGraphErrors(n, t.columns.at("energy").data(), t.columns.at("response").data(),
                                nullptr, t.columns.at("response_err").data());
    for (auto* g : {gr, gl}) {
      g->SetMarkerStyle(kMarker[i % 6]);
      g->SetMarkerColor(kColour[i % 6]);
      g->SetLineColor(kColour[i % 6]);
      g->SetMarkerSize(1.15);
    }
    mgRes.Add(gr, "P");
    mgLin.Add(gl, "P");
    legRes.AddEntry(gr, species[i].second, "p");
    legLin.AddEntry(gl, species[i].second, "p");
  }
  if (mgRes.GetListOfGraphs() == nullptr || mgRes.GetListOfGraphs()->GetSize() == 0) {
    std::cerr << "no curve for " << scenario << "; figure skipped\n";
    return;
  }

  TCanvas cRes((std::string("cResSp_") + scenario).c_str(), "", 800, 600);
  cRes.SetLogx();
  mgRes.Draw("A");
  mgRes.GetXaxis()->SetTitle("E_{true} [GeV]");
  mgRes.GetYaxis()->SetTitle("#sigma(E)/E");
  mgRes.GetYaxis()->SetRangeUser(0.0, 0.30);
  mgRes.GetXaxis()->SetLimits(4.0, 130.0);
  legRes.Draw();
  drawHeader(detail);
  save(cRes, outDir + "/species_resolution_" + scenario);

  TCanvas cLin((std::string("cLinSp_") + scenario).c_str(), "", 800, 600);
  cLin.SetLogx();
  mgLin.Draw("A");
  mgLin.GetXaxis()->SetTitle("E_{true} [GeV]");
  mgLin.GetYaxis()->SetTitle("#LT E_{rec} #GT / E_{true}");
  mgLin.GetYaxis()->SetRangeUser(0.90, 1.12);
  mgLin.GetXaxis()->SetLimits(4.0, 130.0);
  auto* unity = new TLine(4.0, 1.0, 130.0, 1.0);
  unity->SetLineStyle(3);
  unity->SetLineColor(kGray + 2);
  unity->Draw();
  legLin.Draw();
  drawHeader(detail);
  save(cLin, outDir + "/species_linearity_" + scenario);
}

void plotCoupling(const std::string& dir, const std::string& outDir) {
  const std::vector<Scenario> arms = {
      {"pid-none", "no species label"},
      {"pid-bdt", "predicted label"},
      {"pid-true", "true label"},
  };
  for (const char* species : {"all", "pi-", "p"}) {
    const char* label = std::string(species) == "all"    ? "#pi^{-}/p"
                        : std::string(species) == "pi-" ? "#pi^{-}"
                                                        : "protons";
    plotCouplingArm(dir, outDir, "chi2", "#chi^{2} calibration", arms, species, label);
    plotCouplingArm(dir, outDir, "bdt", "BDT regressor", arms, species, label);
  }
  plotCouplingSpecies(dir, outDir, "pid-none_reco-chi2", "#chi^{2} calibration, no species label");
  plotCouplingSpecies(dir, outDir, "pid-none_reco-bdt", "BDT regressor, no species label");
  plotCouplingSpecies(dir, outDir, "pid-bdt_reco-chi2", "#chi^{2} calibration, predicted label");
  plotCouplingSpecies(dir, outDir, "pid-bdt_reco-bdt", "BDT regressor, predicted label");
}

}  // namespace

void plots(const char* kind = "energy") {
  applyStyle();
  const std::string base = outputDir() + "/benchmark/" + kind;
  const std::string dir = base + "/tables";
  const std::string outDir = base + "/plots";
  gSystem->mkdir(outDir.c_str(), kTRUE);

  const std::string what(kind);
  if (what == "energy") {
    plotEnergy(dir, outDir);
  } else if (what == "pid") {
    plotPid(dir, outDir);
  } else if (what == "coupling") {
    plotCoupling(dir, outDir);
  } else {
    std::cerr << "unknown kind '" << what << "'; expected energy, pid or coupling\n";
    return;
  }
  std::cout << "wrote plots to " << outDir << "\n";
}
