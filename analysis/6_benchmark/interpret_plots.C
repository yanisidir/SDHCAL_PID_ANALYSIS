// Redraw the interpretation figures in the style every other benchmark figure
// uses: SDHCAL header, ticks on all four sides, transparent legends.
//
// These three were the last figures drawn by matplotlib. Nothing is recomputed:
// the correlation matrix and the permutation importances are read from the
// tables the interpretation stage already wrote, and the confusion matrix is
// the one published in chapter 6, entered as data so the figure and the text
// cannot drift apart.
//
//   root -l -b -q 'analysis/6_benchmark/interpret_plots.C'

#include "TCanvas.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLatex.h"
#include "TPad.h"
#include "TStyle.h"
#include "TSystem.h"

#include <clocale>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string outputDir() {
  const char* fromEnv = std::getenv("SDHCAL_OUTPUT_DIR");
  return fromEnv ? std::string(fromEnv) : std::string("output");
}

void applyStyle() {
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetPadLeftMargin(0.14);
  gStyle->SetPadBottomMargin(0.13);
  gStyle->SetPadRightMargin(0.04);
  gStyle->SetPadTopMargin(0.06);
  gStyle->SetTitleSize(0.048, "XYZ");
  gStyle->SetLabelSize(0.042, "XYZ");
  gStyle->SetTitleOffset(1.30, "Y");
  gStyle->SetTitleOffset(1.10, "X");
  gStyle->SetTextFont(42);
  gStyle->SetEndErrorSize(3);
  gStyle->SetPaintTextFormat(".1f");
}

void drawHeader(const char* detail) {
  TLatex latex;
  latex.SetNDC();
  // One string rather than two placed by hand: the fixed NDC offset that works
  // at 800 px collides with "Simulation" on a narrower canvas.
  latex.SetTextFont(42);
  latex.SetTextSize(0.048);
  latex.DrawLatex(gPad->GetLeftMargin(), 0.955, "#bf{SDHCAL} #it{Simulation}");
  latex.SetTextFont(42);
  latex.SetTextSize(0.037);
  latex.SetTextAlign(31);
  latex.DrawLatex(1.0 - gPad->GetRightMargin(), 0.955, detail);
}

void save(TCanvas& canvas, const std::string& stem) {
  canvas.SaveAs((stem + ".png").c_str());
  canvas.SaveAs((stem + ".pdf").c_str());
}

std::vector<std::string> splitLine(const std::string& line) {
  std::vector<std::string> cells;
  std::stringstream stream(line);
  std::string cell;
  while (std::getline(stream, cell, ',')) cells.push_back(cell);
  return cells;
}

// ---------------------------------------------------------------------------
void confusionMatrix(const std::string& outDir) {
  // Row-normalised percentages published in chapter 6, true class by row.
  const char* labels[3] = {"#pi^{-}", "p", "K^{0}"};
  const double value[3][3] = {{48.8, 47.9, 3.3}, {27.4, 68.6, 4.0}, {4.0, 10.1, 86.0}};

  auto* h = new TH2F("confusion", "", 3, 0, 3, 3, 0, 3);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) h->SetBinContent(col + 1, 3 - row, value[row][col]);
    h->GetYaxis()->SetBinLabel(3 - row, labels[row]);
    h->GetXaxis()->SetBinLabel(row + 1, labels[row]);
  }
  h->GetXaxis()->SetTitle("classe predite");
  h->GetYaxis()->SetTitle("classe vraie");
  h->GetZaxis()->SetTitle("fraction de la classe vraie [%]");
  h->GetZaxis()->SetTitleSize(0.038);
  h->GetZaxis()->SetTitleOffset(0.95);
  h->GetZaxis()->SetLabelSize(0.032);
  h->GetXaxis()->SetLabelSize(0.055);
  h->GetYaxis()->SetLabelSize(0.055);
  h->SetMinimum(0.0);
  h->SetMaximum(100.0);
  h->SetMarkerSize(2.2);

  TCanvas canvas("cConfusion", "", 760, 640);
  canvas.SetRightMargin(0.16);
  gStyle->SetPalette(kBird);
  h->Draw("COLZ TEXT");
  drawHeader("3 especes, echantillon de test");
  save(canvas, outDir + "/confusion_matrix");
}

// ---------------------------------------------------------------------------
void permutationImportances(const std::string& path, const std::string& outDir) {
  std::ifstream file(path);
  if (!file) {
    std::cerr << "cannot open " << path << "\n";
    return;
  }
  std::string line;
  std::getline(file, line);  // header
  std::vector<std::string> names;
  std::vector<double> importance, spread;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    const std::vector<std::string> cells = splitLine(line);
    if (cells.size() < 3) continue;
    names.push_back(cells[0]);
    importance.push_back(std::atof(cells[1].c_str()));
    spread.push_back(std::atof(cells[2].c_str()));
  }
  const int n = static_cast<int>(names.size());
  if (n == 0) return;

  // Lowest at the bottom, so the strongest variable reads at the top.
  auto* h = new TH1F("importance", "", n, 0, n);
  for (int i = 0; i < n; ++i) {
    const int bin = n - i;  // the file is already sorted, strongest first
    h->SetBinContent(bin, importance[i]);
    h->SetBinError(bin, spread[i]);
    h->GetXaxis()->SetBinLabel(bin, names[i].c_str());
  }
  h->GetYaxis()->SetTitle("chute de l'exactitude equilibree");
  h->GetXaxis()->SetLabelSize(0.030);
  h->SetFillColorAlpha(kAzure + 2, 0.75);
  h->SetLineColor(kAzure + 3);
  h->SetBarWidth(0.75);
  h->SetBarOffset(0.125);

  TCanvas canvas("cImportance", "", 900, 620);
  canvas.SetBottomMargin(0.22);
  h->Draw("BAR E1");
  drawHeader("importance par permutation, #pi^{-}/p");
  save(canvas, outDir + "/permutation_importances");
}

// ---------------------------------------------------------------------------
void correlationMatrix(const std::string& path, const std::string& outDir) {
  std::ifstream file(path);
  if (!file) {
    std::cerr << "cannot open " << path << "\n";
    return;
  }
  std::string line;
  std::getline(file, line);
  const std::vector<std::string> header = splitLine(line);
  std::vector<std::string> names;
  std::vector<std::vector<double>> rows;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    const std::vector<std::string> cells = splitLine(line);
    if (cells.size() < 2) continue;
    names.push_back(cells[0]);
    std::vector<double> row;
    for (size_t i = 1; i < cells.size(); ++i) row.push_back(std::atof(cells[i].c_str()));
    rows.push_back(row);
  }
  const int n = static_cast<int>(names.size());
  if (n == 0) return;

  auto* h = new TH2F("correlation", "", n, 0, n, n, 0, n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n && j < static_cast<int>(rows[i].size()); ++j) {
      h->SetBinContent(j + 1, n - i, rows[i][j]);
    }
    h->GetYaxis()->SetBinLabel(n - i, names[i].c_str());
    h->GetXaxis()->SetBinLabel(i + 1, names[i].c_str());
  }
  h->GetZaxis()->SetTitle("correlation de Pearson");
  h->GetZaxis()->SetTitleSize(0.038);
  h->GetZaxis()->SetTitleOffset(0.95);
  h->GetZaxis()->SetLabelSize(0.032);
  h->SetMinimum(-1.0);
  h->SetMaximum(1.0);
  h->GetXaxis()->SetLabelSize(0.026);
  h->GetYaxis()->SetLabelSize(0.026);
  h->GetXaxis()->LabelsOption("v");

  TCanvas canvas("cCorrelation", "", 900, 800);
  canvas.SetLeftMargin(0.20);
  canvas.SetBottomMargin(0.22);
  canvas.SetRightMargin(0.16);
  // A diverging palette: zero correlation must read as the neutral midpoint.
  gStyle->SetPalette(kLightTemperature);
  h->Draw("COLZ");
  drawHeader("variables d'entree, #pi^{-}/p");
  save(canvas, outDir + "/correlation_matrix");
}

}  // namespace

void interpret_plots() {
  std::setlocale(LC_NUMERIC, "C");  // a comma decimal separator truncates every value
  applyStyle();
  const std::string base = outputDir();
  const std::string outDir = base + "/benchmark/interpret/plots";
  gSystem->mkdir(outDir.c_str(), kTRUE);

  confusionMatrix(outDir);
  permutationImportances(base + "/pid_interpretation/tables/permutation_importances_3.csv", outDir);
  correlationMatrix(base + "/pid_interpretation/tables/correlation_matrix_3.csv", outDir);
  std::cout << "wrote plots to " << outDir << "\n";
}
