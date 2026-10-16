// Usage (batch, no GUI):
// root -l -b -q 'repack_tree.C("130k_kaon_E1to130_params.root","kaon_params_repacked.root","tree")'

#include <memory>
void repack_tree(const char* in="130k_kaon_E1to130_params.root",
                 const char* out="kaon_params_repacked.root",
                 const char* treename="tree") {
  // Avoid ROOT auto-attaching objects to directories (prevents double-deletes)
  TDirectory::AddDirectory(kFALSE);
  gErrorIgnoreLevel = kSysError; // quiet the spam while we rescue data

  std::unique_ptr<TFile> fin(TFile::Open(in, "READ"));
  if (!fin || fin->IsZombie()) { Error("repack_tree","Cannot open input"); return; }

  auto t = dynamic_cast<TTree*>(fin->Get(treename));
  if (!t) { Error("repack_tree","TTree '%s' not found", treename); fin->GetListOfKeys()->Print(); return; }

  // Create clean output and make it current directory
  std::unique_ptr<TFile> fout(TFile::Open(out, "RECREATE"));
  if (!fout || fout->IsZombie()) { Error("repack_tree","Cannot create output"); return; }
  fout->cd();

  // Clone with sorted baskets -> cleaner IO and reduces corruption risk
  TTree* tout = t->CloneTree(-1, "fast SortBasketsByEntry");
  if (!tout) { Error("repack_tree","CloneTree failed"); return; }

  tout->Write();
  fout->Write();
  fout->Close();
  fin->Close();

  printf("Repacked to %s\n", out);
}
