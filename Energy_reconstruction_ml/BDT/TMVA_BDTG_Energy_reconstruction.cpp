// hadron_energy_regressor_TMVA.cpp (C++11, args hardcodés)
// ------------------------------------------------------------
// Entraîne un BDTG TMVA pour régresser l'énergie hadronique
// à partir de paramètres de forme de gerbe. 
// ------------------------------------------------------------
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "TFile.h"
#include "TChain.h"
#include "TTree.h"
#include "TROOT.h"
#include "TMVA/Tools.h"
#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Reader.h"

int main() {
    try {
        // ============================
        // 0) CONFIGURATION HARDCODÉE
        // ============================
        // Liste des fichiers ROOT d'entrée (ici un seul fichier pion)
        // Chaque fichier doit contenir l'arbre avec les features et la vérité
        const std::vector<std::string> inputFiles = {
            "/gridgroup/ilc/midir/analyse/data/merged_primaryEnergy/130k_pi_E1to130_params_merged.root",
        };

        // Nom de l'arbre contenant les données et de la variable cible (label de regression)
        const std::string treeName  = "paramsTree";
        const std::string energyCol = "trueEnergy";

        // Variables explicatives utilisées comme entrées du modèle
        const std::vector<std::string> features = {
            "Thr1", "Thr2", "Thr3", "Radius", "Density", "NClusters",
            "ratioThr23", "Zbary", "Zrms", "PctHitsFirst10",
            "PlanesWithClusmore2", "AvgClustSize", "MaxClustSize",
            "Zbary_thr3", "Zbary_thr2", "tMin", "tMax", "tMean",
            "tSpread", "N1", "N2", "N3"
        };

        // Fichiers de sortie et configuration du modèle TMVA
        const std::string tmvaOutRoot = "TMVA_HadronEnergy.root"; // résultat TMVA
        const std::string datasetName = "dataset";                // dossier interne TMVA
        const std::string methodName  = "BDTG";                   // nom de la méthode
        const std::string weightsXml  = datasetName + "/weights/TMVARegression_" + methodName + ".weights.xml";

        // Hyperparamètres du BDTG (boosted decision tree avec gradient boosting)
        const std::string bdtOptions =
            "!H:!V:"                // désactive histos et verbosité TMVA
            "NTrees=800:"           // nombre d’arbres
            "BoostType=Grad:"       // type de boosting : gradient
            "Shrinkage=0.05:"       // learning rate
            "UseBaggedBoost:"       // bagging activé
            "BaggedSampleFraction=0.8:" // fraction de données par arbre
            "MaxDepth=4:"           // profondeur max des arbres
            "MinNodeSize=2.5%:"     // minimum d’événements par noeud
            "nCuts=20:"             // granularité des coupures
            "SeparationType=RegressionVariance:" // fonction de séparation
            "RegLossFunction=L2:"   // fonction de perte L2 (MSE)
            "IgnoreNegWeightsInTraining"; // ignore poids négatifs

        // ============================================
        // 1) INITIALISATION TMVA
        // ============================================
        TMVA::Tools::Instance(); // singleton TMVA

        // Fichier de sortie TMVA
        std::unique_ptr<TFile> outputFile(TFile::Open(tmvaOutRoot.c_str(), "RECREATE"));
        if (!outputFile || outputFile->IsZombie()) {
            throw std::runtime_error("Impossible de créer le fichier de sortie: " + tmvaOutRoot);
        }

        // Factory = objet principal d'entraînement
        TMVA::Factory factory("TMVARegression",
                              outputFile.get(),
                              "!V:!Silent:Color:DrawProgressBar:AnalysisType=Regression");

        // DataLoader = gestion des données d’entrée
        TMVA::DataLoader loader(datasetName.c_str());

        // Ajout des features et de la cible au DataLoader
        for (size_t i = 0; i < features.size(); ++i) {
            loader.AddVariable(features[i], 'F'); // 'F' = float
        }
        loader.AddTarget(energyCol); // variable cible = énergie vraie

        // ============================================
        // 2) CHARGEMENT DES DONNÉES
        // ============================================
        // TChain = permet d’agréger plusieurs fichiers ROOT
        std::unique_ptr<TChain> chain(new TChain(treeName.c_str()));
        for (size_t i = 0; i < inputFiles.size(); ++i) {
            if (chain->Add(inputFiles[i].c_str()) == 0) {
                std::cerr << "[WARN] Aucun arbre ajouté depuis: " << inputFiles[i] << "\n";
            }
        }
        if (chain->GetEntries() == 0) {
            throw std::runtime_error("Aucun événement dans la TChain (vérifie les fichiers et le nom d'arbre).");
        }

        // On passe la TChain au DataLoader pour la regression
        loader.AddRegressionTree(chain.get());

        // Prépare le split train/test (aléatoire)
        loader.PrepareTrainingAndTestTree("",
            "nTrain_Regression=0:nTest_Regression=0:"   // TMVA choisit auto
            "SplitMode=Random:SplitSeed=42:NormMode=None");

        // ============================================
        // 3) ENTRAÎNEMENT DU BDTG
        // ============================================
        // Déclaration de la méthode de regression
        factory.BookMethod(&loader, TMVA::Types::kBDT, methodName.c_str(), bdtOptions.c_str());

        // Entraînement + test + évaluation automatiques
        factory.TrainAllMethods();
        factory.TestAllMethods();
        factory.EvaluateAllMethods();

        outputFile->Close(); // ferme le fichier résultat

        // =================================================
        // 4) ÉVALUATION MANUELLE SUR L'ÉCHANTILLON TEST
        // =================================================
        TFile resFile(tmvaOutRoot.c_str()); // rouvre fichier TMVA
        if (resFile.IsZombie()) {
            throw std::runtime_error("Impossible de rouvrir le fichier de résultats: " + tmvaOutRoot);
        }

        // Récupère le TestTree généré par TMVA
        TTree* testTree = dynamic_cast<TTree*>(resFile.Get((datasetName + "/TestTree").c_str()));
        if (!testTree) {
            std::cerr << "[ERROR] TestTree introuvable pour l'évaluation: " << datasetName << "/TestTree\n";
            return 1;
        }

        // Prépare un Reader TMVA pour appliquer le modèle entraîné
        std::vector<Float_t> varVals(features.size(), 0.f); // valeurs des features
        Float_t targetVal = 0.f; // vérité terrain (énergie)

        TMVA::Reader reader("!Color:!Silent");
        // Binder les variables/features
        for (size_t i = 0; i < features.size(); ++i) {
            reader.AddVariable(features[i].c_str(), &varVals[i]);
            testTree->SetBranchAddress(features[i].c_str(), &varVals[i]);
        }
        // Bind de la vérité terrain
        testTree->SetBranchAddress(energyCol.c_str(), &targetVal);

        // Charge les poids du modèle entraîné
        if (reader.BookMVA(methodName.c_str(), weightsXml.c_str()) == 0) {
            std::cerr << "[ERROR] Fichier de poids introuvable: " << weightsXml << "\n";
            return 1;
        }

        // Passe 1 : calcul de la moyenne de l’énergie vraie
        Long64_t nTest = testTree->GetEntries();
        if (nTest <= 0) {
            std::cerr << "[ERROR] TestTree vide.\n";
            return 1;
        }

        double sumTrue = 0.0;
        for (Long64_t i = 0; i < nTest; ++i) {
            testTree->GetEntry(i);
            sumTrue += static_cast<double>(targetVal);
        }
        const double meanTrue = sumTrue / static_cast<double>(nTest);

        // Passe 2 : calcul des métriques (RMSE, MAE, R²)
        double ssRes = 0.0, ssTot = 0.0, sumAbs = 0.0;
        for (Long64_t i = 0; i < nTest; ++i) {
            testTree->GetEntry(i);
            // prédiction du modèle
            const double pred = static_cast<double>(reader.EvaluateRegression(methodName.c_str())[0]);
            const double diff = pred - static_cast<double>(targetVal);

            // erreurs cumulées
            ssRes  += diff * diff;           // somme des carrés résiduels
            sumAbs += std::fabs(diff);       // somme des erreurs absolues
            const double dmt = static_cast<double>(targetVal) - meanTrue;
            ssTot  += dmt * dmt;             // somme des carrés totaux
        }

        // Calcul des métriques
        const double rmse = std::sqrt(ssRes / static_cast<double>(nTest));
        const double mae  = sumAbs / static_cast<double>(nTest);
        const double r2   = (ssTot > 0.0) ? (1.0 - ssRes / ssTot) : 1.0;

        // Affiche les résultats
        std::cout << "\n================ Test metrics ================\n"
                  << "Events : " << nTest << "\n"
                  << "RMSE   : " << rmse << "\n"
                  << "MAE    : " << mae  << "\n"
                  << "R^2    : " << r2   << "\n"
                  << "==============================================\n";

        return 0;

    } catch (const std::exception& e) {
        // Gestion des exceptions standard
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
