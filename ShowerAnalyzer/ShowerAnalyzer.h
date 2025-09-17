/**
 * @file ShowerAnalyzer.h
 * @brief Extraction d’observables de gerbes hadroniques à partir de hits digitisés.
 *
 * Cette classe agrège et calcule un ensemble d’objets de “features” pour l’analyse PID
 * et la reconstruction d’énergie à partir des informations par-hit (indices I/J/K, positions,
 * seuils, temps). Elle ne possède pas les données : elle référence des vecteurs externes
 * fournis au constructeur et produit les observables via `analyze()`.
 *
 * Entrées (références constantes) :
 *   - vThr : seuil du hit (1/2/3)
 *   - vK   : indice de couche (plan longitudinal)
 *   - vx, vy, vz : positions (mm) du hit
 *   - vI, vJ : indices de cellule (plans transverses)
 *   - vTime : temps du hit (unité libre, p.ex. ns)
 *
 * Observables calculées (accès via getters) :
 *   - Longitudinal : Zbary, Zrms, Begin (première couche), profil Gaisser–Hillas {Nmax, z0_fit, Xmax, lambda}
 *   - Comptages/ratios : Thr1/Thr2/Thr3, ratioThr23, N1/N2/N3, nHitsTotal
 *   - Morphologie transversale : Radius, Density, eccentricity3D, {lambda1, lambda2}
 *   - Clustering : NClusters, MaxClustSize, AvgClustSize, PlanesWithClusmore2
 *   - Temps : tMin, tMax, tMean, tSpread
 *   - Topologie : NTrackSegments (segments/chaînes adjacentes)
 *   - PctHitsFirst10 : % de hits dans les 10 premiers plans
 *
 * Algorithmes internes (esquisse) :
 *   - Pondération par seuil via `getWeight(thr)` pour certains calculs
 *   - Détection d’adjacence 2D (I,J) intra-couche et construction d’un graphe de clusters
 *   - Longest chain / segments de traces via parcours de graphes
 *   - Ajustements simples et statistiques (barycentre, RMS, SEM…)
 *
 * Utilisation type :
 *   ShowerAnalyzer sa(vThr, vK, vx, vy, vz, vI, vJ, vTime);
 *   sa.analyze();
 *   float zbary = sa.getZbary();  // etc.
 *
 * Remarques :
 *   - L’objet ne gère pas l’allocation des données d’entrée (références non possédées).
 *   - `analyze()` doit être appelé avant lecture des observables.
 *   - Conçu pour un traitement événement-par-événement (non thread-safe par instance partagée).
 */

#ifndef SHOWER_ANALYZER_H
#define SHOWER_ANALYZER_H

#include <vector>
#include <map>
#include <set>
#include <stack>
#include <utility>   // std::pair
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <cstddef>   // size_t

// Déclaration anticipée du type Hit
struct Hit {
    int I, J, K;
    float x, y, z;
    int thr; // utile pour chainLength
};

class ShowerAnalyzer {
public:
    // Constructeur
    ShowerAnalyzer(const std::vector<int>& vThr,
                   const std::vector<int>& vK,
                   const std::vector<float>& vx,
                   const std::vector<float>& vy,
                   const std::vector<float>& vz,
                   const std::vector<int>& vI,
                   const std::vector<int>& vJ,
                   const std::vector<float>& vTime);

    void analyze();  // Lance l’analyse complète

    // Accesseurs pour les observables
    float getZbary() const;
    float getZrms() const;

    float getThr1() const;
    float getThr2() const;
    float getThr3() const;
    float getRatioThr23() const;

    float getRadius() const;
    float getDensity() const;
    int   getBegin() const;

    float getLambda1() const;
    float getLambda2() const;

    int   getN3() const;
    int   getN2() const;
    int   getN1() const;

    int   getNClusters() const;
    int   getMaxClustSize() const;
    float getAvgClustSize() const;
    int   getPlanesWithClusmore2() const;

    float getPctHitsFirst10() const;

    float getTMin()    const;
    float getTMax()    const;
    float getTMean()   const;
    float getTSpread() const;

    // Paramètres du fit Gaisser–Hillas
    float getNmax()   const;
    float getZ0Fit()  const;   // ⚠️ renommé
    float getXmax()   const;
    float getLambda() const;   // ⚠️ renommé

    float getEccentricity3D() const;

    int   getNTrackSegments() const;

private:
    // Entrées
    const std::vector<int>&   vThr_;
    const std::vector<int>&   vK_;
    const std::vector<float>& vx_;
    const std::vector<float>& vy_;
    const std::vector<float>& vz_;
    const std::vector<int>&   vI_;
    const std::vector<int>&   vJ_;
    const std::vector<float>& vTime_;

    // Résultats
    float Zbary_ = 0.0f;
    float Zrms_  = 0.0f;

    float Thr1_ = 0.0f, Thr2_ = 0.0f, Thr3_ = 0.0f;
    float ratioThr23_ = 0.0f;

    float Radius_ = 0.0f;
    float Density_ = 0.0f;
    int   Begin_ = 48;

    float lambda1_ = 0.0f, lambda2_ = 0.0f;

    int   N3_ = 0, N2_ = 0, N1_ = 0;

    int   NClusters_ = 0;
    int   PlanesWithClusmore2_ = 0;
    float AvgClustSize_ = 0.0f;
    int   MaxClustSize_ = 0;

    float PctHitsFirst10_ = 0.0f;

    float tMin_ = 0.0f;
    float tMax_ = 0.0f;
    float tMean_ = 0.0f;
    float tSpread_ = 0.0f;

    // Paramètres du fit Gaisser–Hillas
    float Nmax_   = 0.0f;
    float z0_fit_ = 0.0f;
    float Xmax_   = 0.0f;
    float lambda_ = 0.0f;

    float eccentricity3D_ = 0.0f;

    int   nTrackSegments_ = 0;

    size_t nHitsTotal_ = 0;
    std::vector<Hit> hits;

    // Méthodes privées de calcul
    void computeThresholdPercentages();
    void computeZbaryAndZrms();
    void computeBeginLayer();
    void computeRadius();
    void computeDensity();
    void computeLambdas();
    void computeClusters();
    void computePctHitsFirst10();
    void computeTimeStats();
    void computeLongitudinalProfile();
    void computeEccentricity3D();
    void computeTrackSegments();   

    float getWeight(int thr) const;
    std::vector<std::pair<int,int> > getNeighbors(const std::pair<int,int>& hit) const;

    bool isAdjacent(const Hit& a, const Hit& b) const; 
    int  chainLength(Hit* leaf,
                     const std::unordered_map<Hit*, std::vector<Hit*> >& graph) const; 
};

#endif // SHOWER_ANALYZER_H
