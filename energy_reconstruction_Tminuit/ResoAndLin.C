/*
  ResoAndLin.C
  Superposition des courbes de performance pour π⁻, K⁰_L et p sans minimisation :
  - Résolution relative σ/E (en %) vs E_true
  - Profil de linéarité ⟨E_reco⟩ vs E_true
  - Biais relatif (⟨E_reco⟩ − E_true)/E_true (en %)
  Les trois courbes sont construites à partir d’un modèle Ereco = α*N1 + β*N2 + γ*N3,
  où α, β, γ sont des polynômes quadratiques en Nhit = N1+N2+N3, avec des paramètres
  FIXÉS (ici, cas “réaliste” : on applique les paramètres π à toutes les particules).

  Entrées :
    - digitized : /gridgroup/ilc/midir/analyse/data/digitized/130k_<particle>_E1to130_digitized.root (TTree "tree")
        * thr : std::vector<int> (valeurs 1/2/3)
        * K   : std::vector<int> (indices de couche)
    - raw      : /gridgroup/ilc/midir/analyse/data/raw/130k_<particle>_E1to130.root (TTree "tree")
        * primaryEnergy : double

  Sélection & calcul :
    - exclusion des événements touchant la dernière couche (K == 47)
    - calcul de N1, N2, N3 depuis thr ∈ {1,2,3}
    - Ereco via coefficients (a0..a2, b0..b2, c0..c2) fixes
    - binning commun en énergie vraie : 20 bins entre 10 et 100 GeV
    - σ/E obtenu par fit gaussien de ΔE = Ereco − Etrue dans chaque bin
    - profils de linéarité et biais avec barres d’erreur (SEM)

  Sorties (./plots) :
    - Resolution_relative_pion_to_all_chi2.png   : σ/E (%) vs E_true pour π, K, p
    - Lin_profile_pion_to_all_chi2.png          : ⟨E_reco⟩ vs E_true (profil)
    - Relative_deviation_pion_to_all_chi2.png    : biais relatif (%) vs E_true
    - Lin_n_Dev_pion_to_all_chi2.png             : canvas combiné (linéarité en haut, biais en bas)

  Usage :
    root -l -q 'ResoAndLin.C()'

  Notes :
    - Les paramètres Par9 pour kaon et proton sont ici identiques à ceux du pion (stress test “pion→all”).
    - Les axes Y des figures (résolution/biais) sont réglés pour un zoom lisible (modifiables dans le code).
    - Pas d’appel à Minuit : ce fichier sert à comparer rapidement les courbes multi-particules
      sous un même jeu de paramètres.
*/


#include <vector>
#include <string>
#include <iostream>
#include <cmath>

#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH1F.h"
#include "TLegend.h"
#include "TGraphErrors.h"
#include "TLine.h"
#include "TF1.h"
#include "TROOT.h"
#include "TBox.h"
#include "TPad.h"


// ---------------- Paramètres figés ----------------
struct Par9 { double v[9]; };

// static Par9 PAR_PI = { { // pi
//   4.30851e-02,-3.50665e-05,1.94847e-08,
//   1.07201e-01,-6.36403e-05,1.20235e-08,
//   1.91862e-13,7.35489e-04,-3.00925e-07
// }};

// static Par9 PAR_KAON = { { // kaon
//   5.60298e-02,-4.58223e-05,1.90941e-08,
//   8.00358e-02,-7.76142e-05,4.58240e-08,
//   9.83127e-15,7.97643e-04,-3.56930e-07
// }};

// static Par9 PAR_PROTON = { { // proton
//   4.35267e-02,-1.89745e-05,1.07039e-08,
//   1.22286e-01,-4.93952e-05,5.45074e-09,
//   2.60494e-13,4.77430e-04,-1.68624e-07
// }};

// Cas realiste ou on applique les parammetres du pions pour tt le monde 
static Par9 PAR_PI = { { // pi
  4.30851e-02,-3.50665e-05,1.94847e-08,
  1.07201e-01,-6.36403e-05,1.20235e-08,
  1.91862e-13,7.35489e-04,-3.00925e-07
}};

static Par9 PAR_KAON = { { // kaon
  4.30851e-02,-3.50665e-05,1.94847e-08,
  1.07201e-01,-6.36403e-05,1.20235e-08,
  1.91862e-13,7.35489e-04,-3.00925e-07
}};

static Par9 PAR_PROTON = { { // proton
  4.30851e-02,-3.50665e-05,1.94847e-08,
  1.07201e-01,-6.36403e-05,1.20235e-08,
  1.91862e-13,7.35489e-04,-3.00925e-07
}};

// --------------------------------------------------
// Construit 3 courbes pour une particule :
//  - gRes : sigma/E vs E (0-100 GeV, 20 bins) via fits gaussiens de ΔE
//  - gLin : <E_reco> vs E_true (profil de linéarité)
//  - gDev : ( <E_reco> - E_true ) / E_true vs E_true (déviation relative)
// --------------------------------------------------

// Conversion d'un TGraphErrors en pourcent
static TGraphErrors* ToPercent(const TGraphErrors* g, const std::string& newName) {
    auto* gg = (TGraphErrors*)g->Clone(newName.c_str());
    for (int i=0; i<gg->GetN(); ++i) {
        double x,y; gg->GetPoint(i,x,y);
        gg->SetPoint(i, x, 100.0*y);
        gg->SetPointError(i, gg->GetErrorX(i), 100.0*gg->GetErrorY(i));
    }
    return gg;
}


static bool BuildGraphs(const std::string& particle,
                        const Par9& P,
                        TGraphErrors*& gRes,
                        TGraphErrors*& gLin,
                        TGraphErrors*& gDev)
{
    gRes = 0; gLin = 0; gDev = 0;

    // Fichiers d'entrée (identiques à ton setup)
    const std::string fHitsName   = "/gridgroup/ilc/midir/analyse/data/digitized/130k_" + particle + "_E1to130_digitized.root";
    const std::string fEnergyName = "/gridgroup/ilc/midir/analyse/data/raw/130k_"      + particle + "_E1to130.root";

    TFile* fHits   = TFile::Open(fHitsName.c_str());
    TFile* fEnergy = TFile::Open(fEnergyName.c_str());
    if (!fHits || !fEnergy) { std::cerr << "[BuildGraphs] Erreur open files " << particle << "\n"; return false; }

    TTree* treeHits   = static_cast<TTree*>(fHits->Get("tree"));
    TTree* treeEnergy = static_cast<TTree*>(fEnergy->Get("tree"));
    if (!treeHits || !treeEnergy) { std::cerr << "[BuildGraphs] TTree manquant.\n"; return false; }

    // Branches
    std::vector<int>* vThr = 0;
    std::vector<int>* vK   = 0;
    treeHits->SetBranchAddress("thr", &vThr);
    treeHits->SetBranchAddress("K",   &vK);
    treeHits->SetBranchStatus("*",0);
    treeHits->SetBranchStatus("thr",1);
    treeHits->SetBranchStatus("K",1);

    double primenergy = 0.0;
    treeEnergy->SetBranchAddress("primaryEnergy",   &primenergy);
    treeEnergy->SetBranchStatus("*",0);
    treeEnergy->SetBranchStatus("primaryEnergy",1);
    // Lecture / sélection / calcul Ereco
    const int kLastLayer = 47;

    std::vector<double> Etrue_all; Etrue_all.reserve(200000);
    std::vector<double> Ereco_all; Ereco_all.reserve(200000);

    const Long64_t nEntries = std::min(treeHits->GetEntries(), treeEnergy->GetEntries());
    for (Long64_t i = 0; i < nEntries; ++i) {
        treeHits->GetEntry(i);
        treeEnergy->GetEntry(i);

        bool hasLast = false;
        for (size_t h=0; h<vK->size(); ++h) { if ((*vK)[h] == kLastLayer) { hasLast = true; break; } }
        if (hasLast) continue;
        if (primenergy <= 0.0) continue;

        int N1=0, N2=0, N3=0;
        for (size_t h=0; h<vThr->size(); ++h) {
            const int t = (*vThr)[h];
            if      (t==1) ++N1;
            else if (t==2) ++N2;
            else if (t==3) ++N3;
        }

        const int    Nhit  = N1 + N2 + N3;
        const double alpha = P.v[0] + P.v[1]*Nhit + P.v[2]*Nhit*Nhit;
        const double beta  = P.v[3] + P.v[4]*Nhit + P.v[5]*Nhit*Nhit;
        const double gamma = P.v[6] + P.v[7]*Nhit + P.v[8]*Nhit*Nhit;
        const double Ereco = alpha*N1 + beta*N2 + gamma*N3;

        Etrue_all.push_back(primenergy);
        Ereco_all.push_back(Ereco);
    }

    if (Etrue_all.empty()) { std::cerr << "[BuildGraphs] Pas de données pour " << particle << "\n"; return false; }

    // --------- Binning commun 0–100 GeV ----------
    const int    nBins = 20;
    const double Emin  = 10.0, Emax = 100.0;
    const double dE    = (Emax - Emin)/nBins;

    // ===== 1) Résolution relative : sigma/E =====
    std::vector<double> xR(nBins), yR(nBins), exR(nBins,0.0), eyR(nBins);

    for (int ib=0; ib<nBins; ++ib) {
        const double eLow = Emin + ib*dE;
        const double eHi  = eLow + dE;
        xR[ib] = eLow + 0.5*dE;

        TH1D hde(("hde_"+particle+"_"+std::to_string(ib)).c_str(),
                 "DE;E_{reco}-E_{true} [GeV];N_{events}", 100, -20, 20);

        for (size_t i=0; i<Etrue_all.size(); ++i) {
            const double Et = Etrue_all[i];
            if (Et >= eLow && Et < eHi)
                hde.Fill(Ereco_all[i] - Et);
        }

        hde.Fit("gaus","Q"); // silencieux
        TF1* f = hde.GetFunction("gaus");
        const double sigma = f ? f->GetParameter(2) : 0.0;
        const double esig  = f ? f->GetParError(2)  : 0.0;
        yR[ib]  = (xR[ib] > 0.0) ? sigma / xR[ib] : 0.0;
        eyR[ib] = (xR[ib] > 0.0) ? esig  / xR[ib] : 0.0;
    }

    gRes = new TGraphErrors(nBins, &xR[0], &yR[0], &exR[0], &eyR[0]);
    gRes->SetName(("gRes_"+particle).c_str());
    gRes->SetTitle(";E_{true} [GeV];#sigma/E");

    // ===== 2) Linéarité : <Ereco> =====
    std::vector<double> xL(nBins), yL(nBins), exL(nBins,0.0), eyL(nBins,0.0);

    // ===== 3) Déviation relative =====
    std::vector<double> xD(nBins), yD(nBins), exD(nBins,0.0), eyD(nBins,0.0);

    for (int ib=0; ib<nBins; ++ib) {
        const double eLow = Emin + ib*dE;
        const double eHi  = eLow + dE;
        const double xcen = eLow + 0.5*dE;
        xL[ib] = xcen;
        xD[ib] = xcen;

        // moyenne et écart-type / sqrt(N)
        double sum=0.0, sum2=0.0; int n=0;
        for (size_t i=0; i<Etrue_all.size(); ++i) {
            const double Et = Etrue_all[i];
            if (Et >= eLow && Et < eHi) {
                const double er = Ereco_all[i];
                sum  += er;
                sum2 += er*er;
                ++n;
            }
        }
        if (n>0) {
            const double mean = sum / n;
            const double var  = std::max(0.0, sum2/n - mean*mean);
            const double sem  = std::sqrt(var / n);

            yL[ib]  = mean;
            eyL[ib] = sem;

            if (xcen > 0.0) {
                yD[ib]  = (mean - xcen) / xcen;
                eyD[ib] = sem / xcen;
            } else {
                yD[ib]  = 0.0;
                eyD[ib] = 0.0;
            }
        } else {
            yL[ib]=eyL[ib]=yD[ib]=eyD[ib]=0.0;
        }
    }

    gLin = new TGraphErrors(nBins, &xL[0], &yL[0], &exL[0], &eyL[0]);
    gLin->SetName(("gLin_"+particle).c_str());
    gLin->SetTitle(";E_{true} [GeV];<E_{reco}> [GeV]");

    gDev = new TGraphErrors(nBins, &xD[0], &yD[0], &exD[0], &eyD[0]);
    gDev->SetName(("gDev_"+particle).c_str());
    gDev->SetTitle(";E_{true} [GeV];( <E_{reco}> - E_{true} ) / E_{true}");

    fHits->Close(); fEnergy->Close();
    return true;
}

// --------------------------------------------------
// Macro principale : figures existantes + déviation relative + canvas combiné
// --------------------------------------------------
void ResoAndLin() {
    gSystem->mkdir("plots", kTRUE);

    TGraphErrors *gResPi=nullptr, *gResK=nullptr, *gResP=nullptr;
    TGraphErrors *gLinPi=nullptr, *gLinK=nullptr, *gLinP=nullptr;
    TGraphErrors *gDevPi=nullptr, *gDevK=nullptr, *gDevP=nullptr;

    if (!BuildGraphs("pi",     PAR_PI,     gResPi, gLinPi, gDevPi)) return;
    if (!BuildGraphs("kaon",   PAR_KAON,   gResK,  gLinK,  gDevK )) return;
    if (!BuildGraphs("proton", PAR_PROTON, gResP,  gLinP,  gDevP )) return;

    // ---- Styles
    const int cPi = kRed,      cK = kGreen+2, cP = kBlue;
    const int mPi = 20,        mK = 21,       mP = 22;

    auto style = [](TGraphErrors* g, int col, int m){ g->SetLineColor(col); g->SetMarkerColor(col); g->SetMarkerStyle(m); };

    style(gResPi,cPi,mPi); style(gResK,cK,mK); style(gResP,cP,mP);
    style(gLinPi,cPi,mPi); style(gLinK,cK,mK); style(gLinP,cP,mP);
    style(gDevPi,cPi,mPi); style(gDevK,cK,mK); style(gDevP,cP,mP);

    // ---- Figure 1 : Résolution relative (en %) ----
    TGraphErrors* gResPiPct = ToPercent(gResPi, "gResPiPct");
    TGraphErrors* gResKPct  = ToPercent(gResK,  "gResKPct");
    TGraphErrors* gResPPct  = ToPercent(gResP,  "gResPPct");

    // Styles
    style(gResPiPct,cPi,mPi); style(gResKPct,cK,mK); style(gResPPct,cP,mP);

    // Choisis ton intervalle Y en % (modifiable)
    const double RES_YMIN_PCT = 0.0;   // ex: 0 %
    const double RES_YMAX_PCT = 20.0;  // ex: 30 % (ajuste selon tes données)

    // Cadre avec axe Y en pourcent
    TH1F* frameResPct = new TH1F("frameResPct",";E_{true} [GeV];#sigma/E [%]", 100, 0, 100);
    frameResPct->SetMinimum(RES_YMIN_PCT);
    frameResPct->SetMaximum(RES_YMAX_PCT);
    frameResPct->SetStats(kFALSE);

    TCanvas* cRes = new TCanvas("cResAll","sigma/E (%) vs E (pi,K,p)", 900, 650);
    frameResPct->Draw();
    gResPiPct->Draw("LP SAME");
    gResKPct ->Draw("LP SAME");
    gResPPct ->Draw("LP SAME");

    TLegend* legR = new TLegend(0.60, 0.70, 0.88, 0.88);
    legR->AddEntry(gResPiPct, "#pi^{-}", "lp");
    legR->AddEntry(gResKPct,  "K_{L}^{0}", "lp");
    legR->AddEntry(gResPPct,  "p", "lp");
    legR->Draw();

    // Lignes horizontales repères à 5 % et 10 %
    TLine* lineRes5  = new TLine(0,  5.0, 100,  5.0);
    TLine* lineRes10 = new TLine(0, 10.0, 100, 10.0);
    lineRes5->SetLineColor(kBlack);  lineRes5->SetLineStyle(2);
    lineRes10->SetLineColor(kBlack); lineRes10->SetLineStyle(2);
    lineRes5->Draw("SAME");
    lineRes10->Draw("SAME");

    // cRes->SaveAs("plots/Resolution_relative_all_chi2.png");

    cRes->SaveAs("plots/Resolution_relative_pion_to_all_chi2.png");

    // ---- Figure 2 : Profil de linéarité (inchangé)
    TH1F* frameLin = new TH1F("frameLin",";E_{true} [GeV];<E_{reco}> [GeV]", 100, 0, 100);
    frameLin->SetMinimum(0.0);
    frameLin->SetMaximum(100.0);
    frameLin->SetStats(kFALSE);

    TCanvas* cLin = new TCanvas("cLinAll","Linearity profile (pi,K,p) ", 900, 650);
    frameLin->Draw();
    gLinPi->Draw("P SAME");
    gLinK ->Draw("P SAME");
    gLinP ->Draw("P SAME");

    // diagonale y=x
    TLine* diag = new TLine(0,0,100,100); diag->SetLineColor(kGray+2); diag->SetLineStyle(2); diag->Draw("SAME");

    TLegend* legL = new TLegend(0.55, 0.15, 0.88, 0.32);
    legL->AddEntry(gLinPi, "Linearity #pi^{-}", "lp");
    legL->AddEntry(gLinK,  "Linearity K_{L}^{0}",   "lp");
    legL->AddEntry(gLinP,  "Linearity p",   "lp");
    legL->Draw();

    // cLin->SaveAs("plots/Lin_profile_all_chi2.png");

    cLin->SaveAs("plots/Lin_profile_pion_to_all_chi2.png");

    // ---- Figure 3 : Déviation relative (en %)
    TGraphErrors* gDevPiPct = ToPercent(gDevPi, "gDevPiPct");
    TGraphErrors* gDevKPct  = ToPercent(gDevK,  "gDevKPct");
    TGraphErrors* gDevPPct  = ToPercent(gDevP,  "gDevPPct");

    style(gDevPiPct,cPi,mPi); style(gDevKPct,cK,mK); style(gDevPPct,cP,mP);

    TH1F* frameDevPct = new TH1F("frameDevPct",";E_{true} [GeV];Biais relatif [%]",100,0,100);
    frameDevPct->SetMinimum(-10.0);   // zoom serré ±3 %
    frameDevPct->SetMaximum( 10.0);
    frameDevPct->SetStats(kFALSE);

    TCanvas* cDev = new TCanvas("cDevAll","Relative deviation (%) (pi,K,p) ", 900, 650);
    frameDevPct->Draw();

    // Bande ±1 %
    TBox* band1 = new TBox(0,-1.0,100,1.0);
    band1->SetFillColorAlpha(kGray,0.2);
    band1->SetLineColor(0);
    band1->Draw("SAME");

    gDevPiPct->Draw("LP SAME");
    gDevKPct ->Draw("LP SAME");
    gDevPPct ->Draw("LP SAME");

    // Ligne zéro
    TLine* zero = new TLine(0,0,100,0);
    zero->SetLineColor(kGray+2); zero->SetLineStyle(2); zero->Draw("SAME");

    TLegend* legD = new TLegend(0.60, 0.70, 0.88, 0.88);
    legD->AddEntry(gDevPiPct, "Dev. #pi^{-}", "lp");
    legD->AddEntry(gDevKPct,  "Dev. K_{L}^{0}",   "lp");
    legD->AddEntry(gDevPPct,  "Dev. p",   "lp");
    legD->Draw();

    // cDev->SaveAs("plots/Relative_deviation_all_chi2.png");

    cDev->SaveAs("plots/Relative_deviation_pion_to_all_chi2.png");

    // ---- Figure 4 : Canvas combiné 2 pads (linéarité + déviation)
    TCanvas* cComb = new TCanvas("cComb","Linearity + Deviation", 900, 900);
    TPad* pad1 = new TPad("pad1","",0.0,0.35,1.0,1.0);
    TPad* pad2 = new TPad("pad2","",0.0,0.0, 1.0,0.35);
    pad1->SetBottomMargin(0.0);
    pad2->SetTopMargin(0.0);
    pad2->SetBottomMargin(0.25);
    pad1->Draw(); pad2->Draw();

    // Haut : linéarité
    pad1->cd();
    TH1F* fr1 = (TH1F*)frameLin->Clone("fr1"); fr1->Draw();
    gLinPi->Draw("P SAME");
    gLinK ->Draw("P SAME");
    gLinP ->Draw("P SAME");
    diag->Draw("SAME");

    TLegend* legTop = new TLegend(0.65, 0.15, 0.88, 0.32);
    legTop->AddEntry(gLinPi, "#pi^{-}", "lp");
    legTop->AddEntry(gLinK,  "K_{L}^{0}",   "lp");
    legTop->AddEntry(gLinP,  "p",   "lp");
    legTop->Draw();

    // Bas : déviation relative (en %)
    pad2->cd();
    pad2->SetGridy(true);


    // Graphes en pourcent (on clone pour ne pas modifier les originaux)
    TGraphErrors* gDevPiPct_c = ToPercent(gDevPi, "gDevPiPct_c");
    TGraphErrors* gDevKPct_c  = ToPercent(gDevK,  "gDevKPct_c");
    TGraphErrors* gDevPPct_c  = ToPercent(gDevP,  "gDevPPct_c");

    // Styles identiques au haut
    style(gDevPiPct_c, cPi, mPi);
    style(gDevKPct_c,  cK,  mK);
    style(gDevPPct_c,  cP,  mP);

    // Cadre du bas, en %
    TH1F* fr2 = new TH1F("fr2",";E_{true} [GeV];Biais relatif [%]",100,0,100);
    fr2->SetStats(kFALSE);
    fr2->SetMinimum(-10.0);      // zoom serré ±3 %
    fr2->SetMaximum( 10.0);
    fr2->GetYaxis()->SetNdivisions(505); // ticks plus serrés
    fr2->GetYaxis()->SetNdivisions(505);
    fr2->GetYaxis()->SetLabelSize(0.08);
    fr2->GetYaxis()->SetTitleSize(0.09);
    fr2->GetYaxis()->SetTitleOffset(0.6);
    fr2->GetXaxis()->SetLabelSize(0.08);
    fr2->GetXaxis()->SetTitleSize(0.09);
    fr2->GetXaxis()->SetTitleOffset(0.6);
    fr2->Draw();

    // Bande ±1 % (bas)
    TBox* band1_bot = new TBox(0,-1.0,100,1.0);
    band1_bot->SetFillColorAlpha(kGray,0.15);
    band1_bot->SetLineColor(0);
    band1_bot->Draw("SAME");

    // Courbes
    gDevPiPct_c->Draw("LP SAME");
    gDevKPct_c ->Draw("LP SAME");
    gDevPPct_c ->Draw("LP SAME");

    // Ligne zéro
    TLine* zero_pct = new TLine(0,0,100,0);
    zero_pct->SetLineColor(kGray+2);
    zero_pct->SetLineStyle(2);
    zero_pct->Draw("SAME");

    // cComb->SaveAs("plots/Lin_n_Dev_all_chi2.png");

    cComb->SaveAs("plots/Lin_n_Dev_pion_to_all_chi2.png");

    std::cout << "[OK] Écrit :\n"
              << " - plots/Resolution_relative_all_chi2.png\n"
              << " - plots/Linearity_profile_all_chi2.png\n"
              << " - plots/Relative_deviation_all_chi2.png\n"
              << " - plots/Linearity_and_Deviation_all_chi2.png\n";
}
