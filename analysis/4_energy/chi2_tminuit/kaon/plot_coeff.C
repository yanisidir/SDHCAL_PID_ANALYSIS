#include "TF1.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TAxis.h"

void plot_coeff()
{
    // Création du canevas
    TCanvas *c1 = new TCanvas("c1", "Coefficients Plot for Kaon0", 1000, 600);

    

    // double params[9]{
    // 0.0439714, -5.6607e-05, 3.24703e-08, 0.087253, -9.48706e-05, 5.67426e-08, 1.65017e-14, 0.000919808, -5.26005e-07
    // };

    // double params[9]{
    // 0.0463036, -6.10364e-05, 3.27547e-08, 0.0844443, -0.000109746, 7.18947e-08, 1.05271e-16, 0.000981537, -5.56127e-07
    // };

    double params[9]{
        0.0542081,       // a0
        -4.01026e-05,    // a1
        1.99822e-08,     // a2
        0.0813137,       // b0
        -7.93584e-05,    // b1
        5.69774e-08,     // b2
        4.36006e-14,     // c0
        0.00080654,      // c1
        -4.40469e-07     // c2
    };


    // Définition des fonctions alpha, beta et gamma
    TF1 alpha("alpha", "[0]+[1]*x+[2]*x*x", 0., 1400.); //Usual function
    //TF1 alpha("alpha", "[2] * (x - [1]) * (x - [1]) + [0]", 0., 1400.); //New function
    alpha.SetParameter(0, params[0]);
    alpha.SetParameter(1, params[1]);
    alpha.SetParameter(2, params[2]);
    alpha.SetLineColor(kGreen); // Couleur verte pour alpha
    alpha.SetTitle("Coefficients evolution for Kaon^{0};Number of Hits;Calibration coefficients [GeV]");

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

    c1->SaveAs("root_coeff_kaon.pdf");
}