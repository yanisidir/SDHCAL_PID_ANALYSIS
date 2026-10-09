#include "TF1.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TAxis.h"

void plot_coeff()
{
    // Création du canevas
    TCanvas *c1 = new TCanvas("c1", "Coefficients Plot for Pion-", 1000, 600);

    

    // double params[9]{
    // 0.00742406, 3.46573e-06, 5.3066e-09, 0.204176, -0.000279066, 1.32416e-07, 1.05127e-13, 0.000892811, -4.90758e-07
    // };

    // double params[9]{
    // 0.029354, -3.02819e-05, 2.0329e-08, 0.122824, -0.000145361, 6.45547e-08, 3.664e-13, 0.000895029, -4.70465e-07
    // };

    double params[9]{
        0.0423925,       // a0
        -3.35034e-05,    // a1
        2.41683e-08,     // a2
        0.103441,        // b0
        -4.32675e-05,    // b1
        1.94704e-09,     // b2
        4.24329e-13,     // c0
        0.000736027,     // c1
        -3.70458e-07     // c2
    };

    // Définition des fonctions alpha, beta et gamma
    TF1 alpha("alpha", "[0]+[1]*x+[2]*x*x", 0., 1400.); //Usual function
    //TF1 alpha("alpha", "[2] * (x - [1]) * (x - [1]) + [0]", 0., 1400.); //New function
    alpha.SetParameter(0, params[0]);
    alpha.SetParameter(1, params[1]);
    alpha.SetParameter(2, params[2]);
    alpha.SetLineColor(kGreen); // Couleur verte pour alpha
    alpha.SetTitle("Coefficients evolution for #pi^{-};Number of Hits;Calibration coefficients [GeV]");

    TF1 beta("beta", "[0]+[1]*x+[2]*x*x", 0., 1400.); //Usual function
    //TF1 beta("beta", "[2] * (x - [1]) * (x - [1]) + [0]", 0., 1400.); //New function
    beta.SetParameter(0, params[3]);
    beta.SetParameter(1, params[4]);
    beta.SetParameter(2, params[5]);
    beta.SetLineColor(kBlue); // Couleur bleue pour beta

    TF1 gamma("gamma", "[0]+[1]*x+[2]*x*x", 0., 1400.); //Usual function
    //TF1 gamma("gamma", "[2] * (x - [1]) * (x - [1]) + [0]", 0., 1400.); //New function
    gamma.SetParameter(0, params[6]);
    gamma.SetParameter(1, params[7]);
    gamma.SetParameter(2, params[8]);
    gamma.SetLineColor(kRed); // Couleur rouge pour gamma

    // Dessiner les fonctions
    alpha.Draw(); // Dessiner alpha en premier
    beta.Draw("SAME"); // Dessiner beta sur le même canevas
    gamma.Draw("SAME"); // Dessiner gamma sur le même canevas

    // Ajuster la plage des axes
    alpha.GetXaxis()->SetRangeUser(0, 1400);
    alpha.GetYaxis()->SetRangeUser(-0.1, 0.5); // Ajuster en fonction des valeurs attendues

    // Création de la légende
    TLegend *legend = new TLegend(0.7, 0.7, 0.9, 0.9);
    legend->AddEntry(&alpha, "alpha", "l");
    legend->AddEntry(&beta, "beta", "l");
    legend->AddEntry(&gamma, "gamma", "l");
    legend->Draw();

    // Afficher le canevas
    c1->Update();

    c1->SaveAs("root_coeff_pion.pdf");
}