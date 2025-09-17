// validate_root_file.C
// Diagnostics d'intégrité/cohérence pour: N1,N2,N3,nK,K[nK]/I,primaryEnergy
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TMath.h"

struct Stats { double min=+1e300, max=-1e300; long long neg=0, nan=0, inf=0; };
static void upd(Stats& s, double v){ if(std::isnan(v)) {s.nan++; return;} if(!std::isfinite(v)){s.inf++; return;} s.min=std::min(s.min,v); s.max=std::max(s.max,v); if(v<0) s.neg++; }

void validate_root_file(const char* fname){
  TFile* f = TFile::Open(fname);
  if(!f || f->IsZombie()){ std::cerr<<"[ERR] Cannot open "<<fname<<"\n"; return; }
  TTree* t = (TTree*)f->Get("tree");
  if(!t){ std::cerr<<"[ERR] TTree 'tree' not found\n"; f->Close(); return; }

  // Branch presence
  if(!t->GetBranch("N1") || !t->GetBranch("N2") || !t->GetBranch("N3") ||
     !t->GetBranch("nK") || !t->GetBranch("K")  || !t->GetBranch("primaryEnergy")){
    std::cerr<<"[ERR] Missing one of required branches: N1 N2 N3 nK K primaryEnergy\n";
    f->Close(); return;
  }

  // Attach branches
  Int_t N1=0,N2=0,N3=0,nK=0;
  Double_t E=0.0;
  t->SetBranchAddress("N1",&N1);
  t->SetBranchAddress("N2",&N2);
  t->SetBranchAddress("N3",&N3);
  t->SetBranchAddress("nK",&nK);
  t->SetBranchAddress("primaryEnergy",&E);

  // find nK_max
  Long64_t nent = t->GetEntries();
  int nKmax=0;
  for(Long64_t i=0;i<nent;++i){ t->GetEntry(i); if(nK>nKmax) nKmax=nK; }
  std::vector<Int_t> K(nKmax>0?nKmax:1,0);
  t->SetBranchAddress("K", K.data());

  // Hists
  gSystem->mkdir("plots", kTRUE);
  TH1D hE("hE","primaryEnergy;E [GeV];Events",130,0,130);
  TH1D hNhit("hNhit","Nhit=N1+N2+N3;Nhit;Events",200,0,2000);
  TH1D hN1("hN1","N1;N1;Events",200,0,2000);
  TH1D hN2("hN2","N2;N2;Events",200,0,2000);
  TH1D hN3("hN3","N3;N3;Events",200,0,2000);
  TH1D hK("hK","Layer index K;K;Hits",48, -0.5, 47.5);

  // Counters
  long long cnt_outE=0, cnt_last=0, cnt_badK=0, cnt_badNK=0;
  Stats sE, sN1, sN2, sN3, snK;

  // for correlation
  long double sumE=0, sumN=0, sumEE=0, sumNN=0, sumEN=0;
  long long nCorr=0;

  for(Long64_t i=0;i<nent;++i){
    t->GetEntry(i);
    // basics
    upd(sE,E); upd(sN1,N1); upd(sN2,N2); upd(sN3,N3); upd(snK,nK);
    const int Nhit = N1+N2+N3;
    hE.Fill(E); hNhit.Fill(Nhit); hN1.Fill(N1); hN2.Fill(N2); hN3.Fill(N3);

    // K checks
    if(nK<0){ cnt_badNK++; continue; }
    for(int h=0;h<nK && h<(int)K.size();++h){
      if(K[h]==47) cnt_last++;
      if(K[h]<0 || K[h]>47) cnt_badK++;
      hK.Fill(K[h]);
    }

    // E domain
    if(E<1.0 || E>130.0) cnt_outE++;

    // correlation accumulators (only if finite)
    if(std::isfinite(E) && Nhit>=0){
      sumE += E; sumN += Nhit; sumEE += E*E; sumNN += (long double)Nhit*Nhit; sumEN += (long double)E*Nhit; nCorr++;
    }
  }

  // Pearson corr(E, Nhit)
  double corr = 0.0;
  if(nCorr>1){
    long double num = nCorr*sumEN - sumE*sumN;
    long double den = std::sqrt( (long double)(nCorr*sumEE - sumE*sumE) * (long double)(nCorr*sumNN - sumN*sumN) );
    corr = (den>0)? (double)(num/den) : 0.0;
  }

  // Print report
  std::cout<<"=== VALIDATION REPORT ===\n";
  std::cout<<"File: "<<fname<<"\nEntries: "<<nent<<"\n";
  std::cout<<"primaryEnergy: min="<<sE.min<<" max="<<sE.max<<" neg="<<sE.neg<<" NaN="<<sE.nan<<" Inf="<<sE.inf<<"\n";
  std::cout<<"N1: min="<<sN1.min<<" max="<<sN1.max<<" neg="<<sN1.neg<<"\n";
  std::cout<<"N2: min="<<sN2.min<<" max="<<sN2.max<<" neg="<<sN2.neg<<"\n";
  std::cout<<"N3: min="<<sN3.min<<" max="<<sN3.max<<" neg="<<sN3.neg<<"\n";
  std::cout<<"nK: min="<<snK.min<<" max="<<snK.max<<" neg="<<snK.neg<<"\n";
  std::cout<<"Events with E outside [1,130] GeV: "<<cnt_outE<<"\n";
  std::cout<<"Events with last layer (K==47): "<<cnt_last<<"\n";
  std::cout<<"Bad K values (<0 or >47): "<<cnt_badK<<"\n";
  std::cout<<"Bad nK (<0): "<<cnt_badNK<<"\n";
  std::cout<<"Pearson corr( primaryEnergy , Nhit ) = "<<corr<<"\n";
  std::cout<<"=========================\n";

  // Save plots
  TCanvas c("c","c",900,700);
  c.Clear(); hE.Draw();     c.SaveAs("plots/diag_primaryEnergy.png");
  c.Clear(); hNhit.Draw();  c.SaveAs("plots/diag_Nhit.png");
  c.Clear(); hN1.Draw();    c.SaveAs("plots/diag_N1.png");
  c.Clear(); hN2.Draw();    c.SaveAs("plots/diag_N2.png");
  c.Clear(); hN3.Draw();    c.SaveAs("plots/diag_N3.png");
  c.Clear(); hK.Draw();     c.SaveAs("plots/diag_K.png");

  f->Close();
}

// root -l -q 'validate_root_file.C("/gridgroup/ilc/midir/analyse/data/merged_primaryEnergy/200k_pi-_1-130_params_merged.root")'