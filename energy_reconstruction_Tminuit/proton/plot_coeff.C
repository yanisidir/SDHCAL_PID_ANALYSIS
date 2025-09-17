#include "TF1.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TAxis.h"

void plot_coeff()
{
    // Création du canevas
    TCanvas *c1 = new TCanvas("c1", "Coefficients Plot for Proton", 1000, 600);

    

    // double params[9]{
    // 0.0155734, -2.40907e-06, 6.58112e-09, 0.174905, -0.000147028, 5.07382e-08, 1.3951e-13, 0.00058897, -2.78323e-07
    // };

    // double params[9]{
    // 0.0293418, -1.29286e-05, 7.9884e-09, 0.125128, -9.29261e-05, 3.2017e-08, 3.9586e-13, 0.000564434, -2.47825e-07
    // };

    double params[9]{
        0.0429855,       // a0
        -1.97895e-05,    // a1
        1.45263e-08,     // a2
        0.124109,        // b0
        -4.38829e-05,    // b1
        -3.48943e-10,    // b2
        8.54843e-13,     // c0
        0.000491233,     // c1
        -2.1442e-07      // c2
    };


    // Définition des fonctions alpha, beta et gamma
    TF1 alpha("alpha", "[0]+[1]*x+[2]*x*x", 0., 1400.); //Usual function
    //TF1 alpha("alpha", "[2] * (x - [1]) * (x - [1]) + [0]", 0., 1400.); //New function
    alpha.SetParameter(0, params[0]);
    alpha.SetParameter(1, params[1]);
    alpha.SetParameter(2, params[2]);
    alpha.SetLineColor(kGreen); // Couleur verte pour alpha
    alpha.SetTitle("Coefficients evolution for Proton;Number of Hits;Calibration coefficients [GeV]");

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

    c1->SaveAs("root_coeff_proton.pdf");
}