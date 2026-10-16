void cleanParamsFile() {
  TFile *fin = TFile::Open("130k_proton_E1to130_params.root", "READ");
  TTree *oldTree = (TTree*)fin->Get("paramsTree");
  
  if (!oldTree) {
    Error("cleanParamsFile", "Impossible de trouver l'arbre !");
    return;
  }

  TFile *fout = TFile::Open("130k_proton_E1to130_params_clean.root", "RECREATE");
  TTree *newTree = oldTree->CloneTree(-1);  // clone toutes les entrées

  fout->cd();
  newTree->Write("paramsTree");
  fout->Close();
  fin->Close();
}
