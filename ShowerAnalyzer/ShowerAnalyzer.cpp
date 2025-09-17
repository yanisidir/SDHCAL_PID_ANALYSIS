/**
 * @file ShowerAnalyzer.cpp
 * @brief Implémentation de la classe ShowerAnalyzer : calcul des observables de gerbes
 *        hadroniques (features PID/énergie) à partir des hits digitisés.
 *
 * Contenu :
 *   - Orchestration via `analyze()` qui appelle les modules de calcul :
 *       * computeThresholdPercentages() : N1/N2/N3, %Thr1/2/3, ratioThr23
 *       * computeZbaryAndZrms()        : barycentre longitudinal (Zbary) et Zrms
 *       * computeBeginLayer()           : estimation de la couche de début (quadruplet de couches)
 *       * computeRadius()               : rayon moyen autour d’un axe (fit pol1 sur (z,x) et (z,y), distance)
 *       * computeDensity()              : densité pondérée localement (poids dépendant du seuil)
 *       * computeLambdas()              : λ1, λ2 (VP de la covariance transverse 2×2)
 *       * computeClusters()             : clustering 2D par couche (4-connexité), tailles, compteurs
 *       * computePctHitsFirst10()       : % de hits dans les 10 premiers plans
 *       * computeTimeStats()            : tMin, tMax, tMean, tSpread
 *       * computeLongitudinalProfile()  : graphe longitudinal lissé + fit Gaisser–Hillas (Nmax, z0_fit, Xmax, λ)
 *       * computeEccentricity3D()       : excentricité spatiale 3D (VP de la covariance 3×3)
 *       * computeTrackSegments()        : détection de segments de trace via Hough (xz/yz) et coupes de qualité
 *
 * Détails d’implémentation :
 *   - Adjacence des hits en 6-connexité (I,J,K) et 4/8-connexité intra-couche pour les clusters.
 *   - Poids des seuils : w(1)=1, w(2)=~45.45, w(3)=~136.36 (échelle fC) pour densité/énergie.
 *   - Fit Gaisser–Hillas sur profil longitudinal lissé (moyenne glissante, fenêtre=5).
 *   - Sélection de segments : pics Hough (xz) puis verrouillage (yz), span en K, gap max, résidus moyens,
 *     fraction d’early layers, et nettoyage d’orphelins.
 *
 * Dépendances ROOT utilisées :
 *   TGraph, TF1, TH2I, TMatrixDSym(+Eigen), TVectorD, TMath, TString, Form(), etc.
 *
 * Entrées :
 *   Références vers vThr, vK, vx, vy, vz, vI, vJ, vTime (non possédées par l’objet).
 *   Appeler `analyze()` avant d’accéder aux getters.
 *
 * Sorties (getters principaux) :
 *   Zbary, Zrms, Begin, Radius, Density, λ1/λ2, N1/N2/N3, NClusters/Max/Avg/PlanesWithClusmore2,
 *   PctHitsFirst10, tMin/tMax/tMean/tSpread, Nmax/z0_fit/Xmax/λ, eccentricity3D, nTrackSegments.
 *
 * Remarques :
 *   - Conçu pour un traitement événement-par-événement (une instance par évènement recommandé).
 *   - Certaines constantes (épaisseur de couche, bornes Hough, tailles mini) sont paramétrées en dur
 *     et peuvent être adaptées au détecteur.
 */


#include "ShowerAnalyzer.h"
#include "TGraph.h"
#include "TF1.h"
#include "TROOT.h"    // pour Form
#include "TTree.h"
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"
#include "TString.h"
#include "TVectorD.h"
#include <TH2I.h>
#include <TMath.h>

#include <numeric>   // accumulate
#include <algorithm>
#include <map>
#include <set>
#include <stack>
#include <queue>
#include <unordered_map>
#include <utility>
#include <cmath>
#include <iostream>
#include <vector>
#include <limits>




ShowerAnalyzer::ShowerAnalyzer(const std::vector<int>& vThr,
                               const std::vector<int>& vK,
                               const std::vector<float>& vx,
                               const std::vector<float>& vy,
                               const std::vector<float>& vz,
                               const std::vector<int>& vI,
                               const std::vector<int>& vJ,
                               const std::vector<float>& vTime)
    : vThr_(vThr), vK_(vK), vx_(vx), vy_(vy), vz_(vz), vI_(vI), vJ_(vJ), vTime_(vTime) {}

void ShowerAnalyzer::analyze() {
    nHitsTotal_ = vThr_.size();

    computeThresholdPercentages();
    computeZbaryAndZrms();
    computeBeginLayer();
    computeRadius();
    computeDensity();
    computeLambdas();
    computeClusters();
    computePctHitsFirst10();
    computeTimeStats();
    computeLongitudinalProfile();
    computeEccentricity3D();
    computeTrackSegments();

}



float ShowerAnalyzer::getWeight(int thr) const {
    switch (thr) {
        case 1: return 1.0f;                        // 110 fC
        case 2: return 5000.0f / 110.0f;            // ≈ 45.45
        case 3: return 15000.0f / 110.0f;           // ≈ 136.36
        default: return 0.0f;
    }
}

std::vector<std::pair<int, int>> ShowerAnalyzer::getNeighbors(const std::pair<int, int>& hit) const {
    static const std::vector<std::pair<int, int>> deltas = {
        {0, 1}, {1, 0}, {0, -1}, {-1, 0}
    };
    std::vector<std::pair<int, int>> neighbors;
    for (const auto& d : deltas) {
        neighbors.emplace_back(hit.first + d.first, hit.second + d.second);
    }
    return neighbors;
}

// 6-connectivity in cell-index space
bool ShowerAnalyzer::isAdjacent(const Hit &a, const Hit &b) const {
    int dI = std::abs(a.I - b.I);
    int dJ = std::abs(a.J - b.J);
    int dK = std::abs(a.K - b.K);
    return (dI + dJ + dK == 1);
}

// Compute branch length from a leaf until a junction
int ShowerAnalyzer::chainLength(Hit* leaf, const std::unordered_map<Hit*, std::vector<Hit*>>& graph) const {
    std::queue<Hit*> q;
    std::unordered_map<Hit*, bool> visited;
    q.push(leaf);
    visited[leaf] = true;
    int length = 0;
    while (!q.empty()) {
        Hit* h = q.front(); q.pop();
        length++;
        auto it = graph.find(h);
        if (it == graph.end()) continue;
        for (Hit* nbh : it->second) {
            if (!visited[nbh]) {
                auto it2 = graph.find(nbh);
                if (it2 != graph.end() && it2->second.size() <= 2) {
                    visited[nbh] = true;
                    q.push(nbh);
                }
            }
        }

    }
    return length;
}

void ShowerAnalyzer::computeTimeStats() {
    if (vTime_.empty()) {          // aucun hit → zéros
        tMin_ = tMax_ = tMean_ = tSpread_ = 0.f;
        return;
    }
    auto minmax = std::minmax_element(vTime_.begin(), vTime_.end());
    tMin_ = *minmax.first;
    tMax_ = *minmax.second;
    tSpread_ = tMax_ - tMin_;

    double sum = std::accumulate(vTime_.begin(), vTime_.end(), 0.0);
    tMean_ = static_cast<float>(sum / vTime_.size());
}

void ShowerAnalyzer::computePctHitsFirst10() {
    if (nHitsTotal_ == 0) {
        PctHitsFirst10_ = 0.0f;
        return;
    }

    int count = 0;
    for (int k : vK_) {
        if (k >= 0 && k < 10) {
            ++count;
        }
    }

    PctHitsFirst10_ = static_cast<float>(100.0 * count / nHitsTotal_);
}

void ShowerAnalyzer::computeThresholdPercentages() {
    size_t totalHits = vThr_.size();
    if (totalHits == 0) {
        Thr1_ = Thr2_ = Thr3_ = ratioThr23_ = 0.0f;
        return;
    }

    int count1 = 0, count2 = 0, count3 = 0;
    for (int th : vThr_) {
        switch (th) {
            case 1: ++count1; break;
            case 2: ++count2; break;
            case 3: ++count3; break;
            default: break;
        }
    }
    N1_ = count1;
    N2_ = count2;
    N3_ = count3;
    Thr1_ = static_cast<float>(count1 * 100.0 / totalHits);
    Thr2_ = static_cast<float>(count2 * 100.0 / totalHits);
    Thr3_ = static_cast<float>(count3 * 100.0 / totalHits);
    ratioThr23_ = Thr1_>0 ? static_cast<float>((Thr2_ + Thr3_) / Thr1_) : 0.f;
    
}

void ShowerAnalyzer::computeZbaryAndZrms() {
    if (nHitsTotal_ == 0) {
        Zbary_ = 0.0f;
        Zrms_ = 0.0f;
        return;
    }

    // 1) Calcul de Zbary (barycentre longitudinal)
    double sumZ = 0.0;
    for (int z : vK_) {
        sumZ += static_cast<double>(z);
    }
    Zbary_ = static_cast<float>(sumZ / nHitsTotal_);

    // 2) Calcul du Zrms (écart-type)
    double sumSq = 0.0;
    for (int z : vK_) {
        double dz = static_cast<double>(z) - Zbary_;
        sumSq += dz * dz;
    }
    Zrms_ = static_cast<float>(std::sqrt(sumSq / nHitsTotal_));
}


void ShowerAnalyzer::computeBeginLayer() {
    // 1) Compter les hits par couche
    std::map<int, int> hitsPerLayer;
    for (int k : vK_) {
        hitsPerLayer[k]++;
    }

    // 2) Extraire et trier les couches
    std::vector<int> layers;
    layers.reserve(hitsPerLayer.size());
    for (const auto& p : hitsPerLayer) {
        layers.push_back(p.first);
    }
    std::sort(layers.begin(), layers.end());

    // 3) Chercher un quadruplet consécutif valide
    Begin_ = 48;  // valeur par défaut
    for (size_t i = 0; i + 3 < layers.size(); ++i) {
        int L = layers[i];
        bool valid = true;
        for (int d = 0; d < 4; ++d) {
            if (hitsPerLayer[L + d] < 4) {
                valid = false;
                break;
            }
        }
        if (valid) {
            Begin_ = L;
            break;
        }
    }
}

void ShowerAnalyzer::computeRadius() {
    // 1) Regrouper les hits par couche
    std::map<int, std::vector<size_t>> hitsByLayer;
    for (size_t i = 0; i < vK_.size(); ++i) {
        hitsByLayer[vK_[i]].push_back(i);
    }

    // 2) Trier les couches et garder les 10 premières
    std::vector<int> layers;
    for (const auto& p : hitsByLayer) layers.push_back(p.first);
    std::sort(layers.begin(), layers.end());

    size_t nFitLayers = std::min<size_t>(10, layers.size());
    if (nFitLayers < 2) {
        Radius_ = 0.0f;
        return;
    }

    std::vector<double> xFit, yFit, zFit;

    for (size_t i = 0; i < nFitLayers; ++i) {
        int layer = layers[i];
        const auto& indices = hitsByLayer[layer];

        double sumX = 0.0, sumY = 0.0, sumZ = 0.0;
        for (size_t idx : indices) {
            sumX += vx_[idx];
            sumY += vy_[idx];
            sumZ += vz_[idx];
        }
        size_t n = indices.size();
        xFit.push_back(sumX / n);
        yFit.push_back(sumY / n);
        zFit.push_back(sumZ / n);
    }

    // 3) Fit des droites
    TGraph gx(static_cast<int>(zFit.size()), &zFit[0], &xFit[0]);
    TGraph gy(static_cast<int>(zFit.size()), &zFit[0], &yFit[0]);
    TF1 fx("fx", "pol1"), fy("fy", "pol1");
    gx.Fit(&fx, "Q");  // fit silencieux
    gy.Fit(&fy, "Q");

    double a1 = fx.GetParameter(1), b1 = fx.GetParameter(0);
    double a2 = fy.GetParameter(1), b2 = fy.GetParameter(0);
    double denom = a1 * a1 + a2 * a2 + 1.0;

    // 4) Distance de chaque hit à l’axe estimé
    double sumd2 = 0.0;
    for (size_t i = 0; i < vx_.size(); ++i) {
        double xi = vx_[i], yi = vy_[i], zi = vz_[i];

        double t = (a1 * xi + a2 * yi + zi - a1 * b1 - a2 * b2) / denom;
        double x0 = a1 * t + b1;
        double y0 = a2 * t + b2;
        double dx = xi - x0;
        double dy = yi - y0;
        sumd2 += dx * dx + dy * dy;
    }

    Radius_ = static_cast<float>(std::sqrt(sumd2 / vx_.size()));
}

void ShowerAnalyzer::computeDensity() {
    size_t nHits = vThr_.size();
    if (nHits == 0) {
        Density_ = 0.0f;
        return;
    }

    // 1) Calcul du poids total
    double totalWeight = 0.0;
    for (size_t i = 0; i < nHits; ++i) {
        totalWeight += getWeight(vThr_[i]);
    }

    if (totalWeight == 0.0) {
        Density_ = 0.0f;
        return;
    }

    // 2) Densité locale pour chaque hit
    std::vector<double> localCounts(nHits, 0.0);

    for (size_t i = 0; i < nHits; ++i) {
        int ki = vK_[i];
        int ji = vJ_[i];
        int ii = vI_[i];

        double localSum = 0.0;
        for (size_t j = 0; j < nHits; ++j) {
            if (vK_[j] == ki &&
                std::abs(vJ_[j] - ji) <= 1 &&
                std::abs(vI_[j] - ii) <= 1) {
                localSum += getWeight(vThr_[j]);
            }
        }

        localCounts[i] = localSum;
    }

    // 3) Moyenne des densités locales pondérées
    double sumLocal = 0.0;
    for (double c : localCounts) {
        sumLocal += c;
    }

    Density_ = static_cast<float>(sumLocal / totalWeight);
}

void ShowerAnalyzer::computeLambdas() {
    if (nHitsTotal_ == 0) {
        lambda1_ = lambda2_ = 0.0f;
        return;
    }

    // 1) Barycentre
    double sumX = 0.0, sumY = 0.0;
    for (size_t i = 0; i < vx_.size(); ++i) {
        sumX += vx_[i];
        sumY += vy_[i];
    }
    double x0 = sumX / nHitsTotal_;
    double y0 = sumY / nHitsTotal_;

    // 2) Matrice de covariance
    double cxx = 0.0, cxy = 0.0, cyy = 0.0;
    for (size_t i = 0; i < vx_.size(); ++i) {
        double dx = vx_[i] - x0;
        double dy = vy_[i] - y0;
        cxx += dx * dx;
        cxy += dx * dy;
        cyy += dy * dy;
    }
    cxx /= nHitsTotal_;
    cxy /= nHitsTotal_;
    cyy /= nHitsTotal_;

    // 3) Valeurs propres de la matrice 2x2 symétrique
    double trace = cxx + cyy;
    double det = cxx * cyy - cxy * cxy;
    double discriminant = std::sqrt(std::max(0.0, 0.25 * trace * trace - det));

    double l1 = 0.5 * trace + discriminant;
    double l2 = 0.5 * trace - discriminant;

    if (l1 >= l2) {
        lambda1_ = static_cast<float>(l1);
        lambda2_ = static_cast<float>(l2);
    } else {
        lambda1_ = static_cast<float>(l2);
        lambda2_ = static_cast<float>(l1);
    }
}

void ShowerAnalyzer::computeClusters() {
    // 1) Regrouper les hits par plan k
    std::map<int, std::set<std::pair<int, int>>> hitsByPlane;
    for (size_t idx = 0; idx < vK_.size(); ++idx) {
        hitsByPlane[vK_[idx]].emplace(vI_[idx], vJ_[idx]);
    }

    std::vector<int> allClusterSizes;
    PlanesWithClusmore2_ = 0;  

    // 2) Parcours par plan (sans structured binding)
    for (auto& kv : hitsByPlane) {
        auto& hitSet    = kv.second;
        std::vector<int> sizesThisPlane;

        while (!hitSet.empty()) {
            // DFS pour explorer le cluster
            std::stack<std::pair<int, int>> stack;
            stack.push(*hitSet.begin());
            hitSet.erase(hitSet.begin());
            int clusterSize = 1;

            while (!stack.empty()) {
                auto current = stack.top();
                stack.pop();

                for (const auto& neighbor : getNeighbors(current)) {
                    auto it = hitSet.find(neighbor);
                    if (it != hitSet.end()) {
                        stack.push(*it);
                        hitSet.erase(it);
                        ++clusterSize;
                    }
                }
            }

            sizesThisPlane.push_back(clusterSize);
            allClusterSizes.push_back(clusterSize);
        }

        if (sizesThisPlane.size() >= 2) {
            ++PlanesWithClusmore2_;
        }
    }

    // 3) Résultats finaux
    NClusters_ = static_cast<int>(allClusterSizes.size());

    MaxClustSize_ = 0;
    double sumSizes = 0.0;
    for (int sz : allClusterSizes) {
        sumSizes += sz;
        if (sz > MaxClustSize_) {
            MaxClustSize_ = sz;
        }
    }

    AvgClustSize_ = (NClusters_ > 0)
                  ? static_cast<float>(sumSizes / NClusters_)
                  : 0.0f;
}

void ShowerAnalyzer::computeLongitudinalProfile() {

    // 0) Nettoyage
    hits.clear();
    hits.reserve(vx_.size());

    // 1) Lecture des hits
    for (size_t idx = 0; idx < vx_.size(); ++idx) {
        Hit hh;
        hh.I   = vI_[idx];
        hh.J   = vJ_[idx];
        hh.K   = vK_[idx];
        hh.x   = vx_[idx];
        hh.y   = vy_[idx];
        hh.z   = vz_[idx];
        hh.thr = static_cast<double>(vThr_[idx]);
        hits.push_back(hh);
    }


    // 2) On a besoin d'au moins 5 hits
    if (hits.size() < 5) return;


    // Build 6-connectivity graph in index space
    std::unordered_map<Hit*, std::vector<Hit*>> graph;
    graph.reserve(hits.size());
    for (size_t i = 0; i < hits.size(); ++i) {
        for (size_t j = i + 1; j < hits.size(); ++j) {
            if (isAdjacent(hits[i], hits[j])) {
                graph[&hits[i]].push_back(&hits[j]);
                graph[&hits[j]].push_back(&hits[i]);
            }
        }
    }

    // Prune solitary branches
    const int MIN_CHAIN_LENGTH = 5;
    const double ENERGY_THRESHOLD = 0.1;
    auto pruneOnce = [&]() {
        bool removed = false;
        for (auto it = graph.begin(); it != graph.end(); ) {
            Hit* node = it->first;
            auto& neigh = it->second;
            if (neigh.size() == 1) {
                if (chainLength(node, graph) < MIN_CHAIN_LENGTH || node->thr < ENERGY_THRESHOLD) {
                    Hit* parent = neigh[0];
                    auto &pne = graph[parent];
                    pne.erase(std::remove(pne.begin(), pne.end(), node), pne.end());
                    it = graph.erase(it);
                    removed = true;
                    continue;
                }
            }
            ++it;
        }
        return removed;
    };
    while (pruneOnce());

    // Group remaining hits by layer K
    std::map<int, std::vector<Hit*>> hitsByLayer;
    for (auto &kv : graph) {
        hitsByLayer[kv.first->K].push_back(kv.first);
    }

    // Find true start layer: >=3 hits for 4 consecutive layers
    const int MIN_HITS_PER_LAYER = 3;
    const int CONSECUTIVE_LAYERS = 4;
    int startLayer = -1;
    for (auto it = hitsByLayer.begin(); it != hitsByLayer.end(); ++it) {
        int base = it->first;
        bool ok = true;
        for (int j = 0; j < CONSECUTIVE_LAYERS; ++j) {
            if (hitsByLayer[base+j].size() < MIN_HITS_PER_LAYER) { ok = false; break; }
        }
        if (ok) { startLayer = base; break; }
    }
    if (startLayer < 0 && !hitsByLayer.empty()) startLayer = hitsByLayer.begin()->first;

    // Build longitudinal profile (layer → sum(thr))
    const double layerThickness = 10.0; // mm, adapter à ton détecteur
    const int MAX_LAYER = 29;
    std::vector<double> zvec, Evec;
    for (auto &kv : hitsByLayer) {
        int layer = kv.first;
        if (layer < startLayer || layer > MAX_LAYER) continue;
        double sumThr = 0;
        for (Hit* ph : kv.second) sumThr += ph->thr;
        zvec.push_back(layer * layerThickness);
        Evec.push_back(sumThr);
    }
    if (zvec.size() < 5) {
        std::cerr << "Not enough points (" << zvec.size() << ")\n";
        return;
    }

    // Smooth (moving average, window=5)
    std::vector<double> E_smooth(Evec.size());
    int w = 2;
    for (size_t i = 0; i < Evec.size(); ++i) {
        int L = std::max<int>(0, i - w);
        int U = std::min<int>(Evec.size()-1, i + w);
        double s = 0;
        for (int j = L; j <= U; ++j) s += Evec[j];
        E_smooth[i] = s / (U - L + 1);
    }

    // --- 4) construction du TGraph et fit Gaisser–Hillas ---
    TGraph *gr = new TGraph(zvec.size(), &zvec[0], &E_smooth[0]);
    gr->SetName(Form("longProf_evt%lu", static_cast<unsigned long>(nHitsTotal_)));
    gr->SetTitle(Form("Profil longitudinal;profondeur [mm];sum(thr)"));

    TF1 *gh = new TF1("gh",
        "[0]*pow((x-[1])/( [2]-[1] ),(( [2]-[1] )/[3]))*exp((([2]-x)/[3]))",
        zvec.front(), zvec.back());
    gh->SetParameters(*std::max_element(E_smooth.begin(), E_smooth.end()),
                      zvec.front(), zvec[zvec.size()/2], 10);
    gh->SetParNames("Nmax","z0","Xmax","lambda");
    gr->Fit("gh","Q");

    // 5) Stockage des résultats
    Nmax_   = gh->GetParameter(0);
    z0_fit_ = gh->GetParameter(1);
    Xmax_   = gh->GetParameter(2);
    lambda_ = gh->GetParameter(3);


}

void ShowerAnalyzer::computeEccentricity3D() {
    // Excentricité basée sur la répartition spatiale physique (x,y,z)
    // Définition : eig_max / (eig_x + eig_y + eig_z) où eig_* sont les VP de la covariance 3x3

    if (nHitsTotal_ < 2) { 
        eccentricity3D_ = 0.f; 
        return; 
    }

    // 1) Barycentre en (x,y,z)
    double mx = 0.0, my = 0.0, mz = 0.0;
    for (size_t i = 0; i < vx_.size(); ++i) {
        mx += vx_[i];
        my += vy_[i];
        mz += vz_[i];
    }
    mx /= static_cast<double>(nHitsTotal_);
    my /= static_cast<double>(nHitsTotal_);
    mz /= static_cast<double>(nHitsTotal_);

    // 2) Matrice de covariance 3×3 sur (x,y,z)
    TMatrixDSym cov(3);
    cov.Zero();
    for (size_t i = 0; i < vx_.size(); ++i) {
        const double dx = static_cast<double>(vx_[i]) - mx;
        const double dy = static_cast<double>(vy_[i]) - my;
        const double dz = static_cast<double>(vz_[i]) - mz;

        cov(0,0) += dx*dx; cov(0,1) += dx*dy; cov(0,2) += dx*dz;
        cov(1,1) += dy*dy; cov(1,2) += dy*dz;
        cov(2,2) += dz*dz;
    }
    cov(1,0) = cov(0,1); 
    cov(2,0) = cov(0,2); 
    cov(2,1) = cov(1,2);

    // Normalisation par N (cohérent avec le reste de ta classe)
    cov *= (1.0 / static_cast<double>(nHitsTotal_));

    // 3) Valeurs propres (ordre croissant)
    TMatrixDSymEigen eigenSolver(cov);
    const TVectorD& eig = eigenSolver.GetEigenValues();

    const double sum = eig(0) + eig(1) + eig(2);
    eccentricity3D_ = (sum > 0.0) 
        ? static_cast<float>(eig(2) / sum)   // plus grande VP / somme
        : 0.f;
}


void ShowerAnalyzer::computeTrackSegments() {
    // 1) Construire les hits depuis les vecteurs internes
    std::vector<Hit> localHits;
    localHits.reserve(vK_.size());
    for (size_t i = 0; i < vK_.size(); ++i) {
        Hit h;
        h.I = vI_[i]; h.J = vJ_[i]; h.K = vK_[i];
        h.x = vx_[i]; h.y = vy_[i]; h.z = vz_[i];
        localHits.push_back(h);
    }

    // 2) Clustering 2D (I,J) par couche K (8-connexité)
    const int nLayersMax = 50;
    std::vector<std::vector<Hit> > layerHits(nLayersMax);
    for (const auto &h : localHits) {
        if (0 <= h.K && h.K < nLayersMax) layerHits[h.K].push_back(h);
    }

    struct Cluster { std::vector<Hit> hits; double xc, yc, zc; };
    std::vector<Cluster> allClusters;
    allClusters.reserve(localHits.size()); // majoration

    for (int lay = 0; lay < nLayersMax; ++lay) {
        std::vector<Hit> &vh = layerHits[lay];
        std::vector<bool> used(vh.size(), false);
        for (size_t i = 0; i < vh.size(); ++i) if (!used[i]) {
            Cluster cl;
            std::vector<int> stack(1, static_cast<int>(i));
            used[i] = true;
            while (!stack.empty()) {
                int idx = stack.back(); stack.pop_back();
                cl.hits.push_back(vh[idx]);
                for (size_t j = 0; j < vh.size(); ++j) if (!used[j]) {
                    if (std::abs(vh[j].I - vh[idx].I) <= 1 &&
                        std::abs(vh[j].J - vh[idx].J) <= 1) {
                        used[j] = true;
                        stack.push_back(static_cast<int>(j));
                    }
                }
            }
            double sx = 0.0, sy = 0.0, sz = 0.0;
            for (const Hit &hh : cl.hits) { sx += hh.x; sy += hh.y; sz += hh.z; }
            const double inv = 1.0 / static_cast<double>(cl.hits.size());
            cl.xc = sx * inv; cl.yc = sy * inv; cl.zc = sz * inv;
            allClusters.push_back(cl);
        }
    }

    // 3) Filtrer clusters trop gros / zones trop denses
    std::vector<Cluster> selClusters;
    selClusters.reserve(allClusters.size());
    for (size_t ii = 0; ii < allClusters.size(); ++ii) {
        Cluster &ci = allClusters[ii];
        if (static_cast<int>(ci.hits.size()) > 15) continue;
        int nNeigh = 0, largeNeigh = 0;
        for (size_t jj = 0; jj < allClusters.size(); ++jj) {
            if (ii == jj) continue;
            Cluster &cj = allClusters[jj];
            if (std::abs(cj.xc - ci.xc) < 50.0 && std::abs(cj.yc - ci.yc) < 50.0) {
                ++nNeigh;
                if (static_cast<int>(cj.hits.size()) > 5) ++largeNeigh;
            }
        }
        if (nNeigh > 4 || largeNeigh > 2) continue;
        selClusters.push_back(ci);
    }

    // 4) Hough (z,x) pour directions candidates
    const int nTh = 100, nRo = 100;
    const double maxRho = 1500.0;
    TH2I h2xz("h2xz", ";#theta [rad];#rho [mm]", nTh, 0, TMath::Pi(), nRo, 0, maxRho);
    for (size_t ic = 0; ic < selClusters.size(); ++ic) {
        Cluster &c = selClusters[ic];
        for (int ib = 1; ib <= nTh; ++ib) {
            const double theta = h2xz.GetXaxis()->GetBinCenter(ib);
            const double rho   = c.zc * TMath::Cos(theta) + c.xc * TMath::Sin(theta);
            if (rho >= 0.0 && rho < maxRho) h2xz.Fill(theta, rho);
        }
    }

    // Helpers
    std::vector<bool> usedCluster(selClusters.size(), false);
    const int    kBins = 2;

    const int spanK_min   = 6;
    const int maxGapK_max = 3;

    const int nTracksMinIdx = 5; // min clusters per segment

    // lambdas utilitaires
    std::vector<int> tmpKs;
    tmpKs.reserve(selClusters.size());

    const double dRoX = (kBins + 0.5) * h2xz.GetYaxis()->GetBinWidth(1);

    // fonctions utilitaires
    const std::vector<int> emptyIdxs;
    (void)emptyIdxs;

    // span K
    const auto spanK_for = [&](const std::vector<int>& idxs)->int{
        if (idxs.empty()) return 0;
        int kmin =  std::numeric_limits<int>::max();
        int kmax = -std::numeric_limits<int>::max();
        for (size_t u = 0; u < idxs.size(); ++u) {
            int k = selClusters[idxs[u]].hits.front().K;
            if (k < kmin) kmin = k;
            if (k > kmax) kmax = k;
        }
        return (kmax - kmin + 1);
    };

    const auto maxGapK_for = [&](const std::vector<int>& idxs)->int{
        if (idxs.size() <= 1) return 0;
        tmpKs.clear(); tmpKs.reserve(idxs.size());
        for (size_t u = 0; u < idxs.size(); ++u)
            tmpKs.push_back(selClusters[idxs[u]].hits.front().K);
        std::sort(tmpKs.begin(), tmpKs.end());
        int mg = 0;
        for (size_t i = 1; i < tmpKs.size(); ++i) mg = std::max(mg, tmpKs[i] - tmpKs[i-1]);
        return mg;
    };

    const auto meanAbsResidualXZ = [&](double theta,double rho,const std::vector<int>& idxs)->double{
        if (idxs.empty()) return 1e9;
        double s = 0.0; int n = 0;
        for (size_t u = 0; u < idxs.size(); ++u) {
            Cluster &c = selClusters[idxs[u]];
            const double r = c.zc * TMath::Cos(theta) + c.xc * TMath::Sin(theta);
            s += std::abs(r - rho); ++n;
        }
        return s / n;
    };

    const auto meanAbsResidualYZ = [&](double theta,double rho,const std::vector<int>& idxs)->double{
        if (idxs.empty()) return 1e9;
        double s = 0.0; int n = 0;
        for (size_t u = 0; u < idxs.size(); ++u) {
            Cluster &c = selClusters[idxs[u]];
            const double r = c.zc * TMath::Cos(theta) + c.yc * TMath::Sin(theta);
            s += std::abs(r - rho); ++n;
        }
        return s / n;
    };

    const auto earlyFrac_for = [&](const std::vector<int>& idxs,int kEarlyMax)->double{
        if (idxs.empty()) return 0.0;
        int early = 0;
        for (size_t u = 0; u < idxs.size(); ++u) {
            int k = selClusters[idxs[u]].hits.front().K;
            if (k <= kEarlyMax) ++early;
        }
        return 100.0 * early / static_cast<double>(idxs.size());
    };

    // 5) Recherche des maxima locaux en (theta,rho) et sélection finale
    int nTracks = 0;

    for (int ibx = 2; ibx < nTh; ++ibx) {
        for (int iby = 2; iby < nRo; ++iby) {
            int val = h2xz.GetBinContent(ibx, iby);
            if (val <= 2) continue;

            bool isMax = true;
            for (int dx = -1; dx <= 1 && isMax; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    if (dx == 0 && dy == 0) continue;
                    if (h2xz.GetBinContent(ibx+dx, iby+dy) > val) { isMax = false; break; }
                }
            }
            if (!isMax) continue;

            const double theta0 = h2xz.GetXaxis()->GetBinCenter(ibx);
            const double rho0   = h2xz.GetYaxis()->GetBinCenter(iby);

            // Fenêtre en rho (xz)
            std::vector<int> idxSegment;
            idxSegment.reserve(selClusters.size());
            for (size_t ic = 0; ic < selClusters.size(); ++ic) {
                if (usedCluster[ic]) continue;
                Cluster &c = selClusters[ic];
                const double r = c.zc * TMath::Cos(theta0) + c.xc * TMath::Sin(theta0);
                if (std::abs(r - rho0) <= dRoX) idxSegment.push_back(static_cast<int>(ic));
            }
            if (static_cast<int>(idxSegment.size()) < nTracksMinIdx) continue;

            // 2e Hough (z,y) pour verrouiller la direction 3D
            TH2I h2yz("h2yz", ";#theta';#rho'", nTh, 0, TMath::Pi(), nRo, 0, maxRho);
            for (size_t u = 0; u < idxSegment.size(); ++u) {
                Cluster &c = selClusters[idxSegment[u]];
                for (int ib = 1; ib <= nTh; ++ib) {
                    const double t = h2yz.GetXaxis()->GetBinCenter(ib);
                    const double r = c.zc * TMath::Cos(t) + c.yc * TMath::Sin(t);
                    if (r >= 0.0 && r < maxRho) h2yz.Fill(t, r);
                }
            }
            bool   found3D = false;
            double theta1 = 0.0, rho1 = 0.0;
            int best = 0, bestx = -1, besty = -1;
            for (int i2x = 2; i2x < nTh; ++i2x) {
                for (int i2y = 2; i2y < nRo; ++i2y) {
                    int c = h2yz.GetBinContent(i2x, i2y);
                    if (c > best) { best = c; bestx = i2x; besty = i2y; }
                }
            }
            if (best > 4) { found3D = true; theta1 = h2yz.GetXaxis()->GetBinCenter(bestx); rho1 = h2yz.GetYaxis()->GetBinCenter(besty); }
            if (!found3D) continue;

            // Orphelins (exige >=1 voisin en K±1)
            std::vector<int> finalIdx;
            finalIdx.reserve(idxSegment.size());
            for (size_t uu = 0; uu < idxSegment.size(); ++uu) {
                int ic = idxSegment[uu];
                int layC = selClusters[ic].hits.front().K;
                int cnt = 0;
                for (size_t vv = 0; vv < idxSegment.size(); ++vv) {
                    if (uu == vv) continue;
                    int jc = idxSegment[vv];
                    int layJ = selClusters[jc].hits.front().K;
                    if (std::abs(layJ - layC) <= 1) ++cnt;
                }
                if (cnt >= 1) finalIdx.push_back(ic);
            }
            if (static_cast<int>(finalIdx.size()) < nTracksMinIdx) continue;

            // Coupures de qualité
            const int spanK   = spanK_for(finalIdx);           if (spanK < spanK_min) continue;
            const int maxGapK = maxGapK_for(finalIdx);         if (maxGapK > maxGapK_max) continue;
            const double meanResXZ = meanAbsResidualXZ(theta0, rho0, finalIdx);
            const double dRoY      = (kBins + 0.5) * h2yz.GetYaxis()->GetBinWidth(1);
            const double meanResYZ = meanAbsResidualYZ(theta1, rho1, finalIdx);
            if ( (meanResXZ > 1.5 * dRoX) || (meanResYZ > 1.5 * dRoY) ) continue;
            const double earlyPct  = earlyFrac_for(finalIdx, 8); if (earlyPct > 60.0) continue;

            // Segment validé
            ++nTracks;
            for (size_t u = 0; u < finalIdx.size(); ++u) usedCluster[finalIdx[u]] = true;
        }
    }

    // 6) Résultat
    nTrackSegments_ = nTracks;
}

float ShowerAnalyzer::getThr1() const { return Thr1_; }
float ShowerAnalyzer::getThr2() const { return Thr2_; }
float ShowerAnalyzer::getThr3() const { return Thr3_; }
float ShowerAnalyzer::getRatioThr23() const { return ratioThr23_; }

int ShowerAnalyzer::getN3() const { return N3_; }
int ShowerAnalyzer::getN2() const { return N2_; }
int ShowerAnalyzer::getN1() const { return N1_; }

float ShowerAnalyzer::getZbary() const { return Zbary_; }
float ShowerAnalyzer::getZrms() const { return Zrms_; }

int ShowerAnalyzer::getBegin() const { return Begin_; }
float ShowerAnalyzer::getRadius() const { return Radius_; }
float ShowerAnalyzer::getDensity() const { return Density_; }
float ShowerAnalyzer::getLambda1() const { return lambda1_; }
float ShowerAnalyzer::getLambda2() const { return lambda2_; }

int ShowerAnalyzer::getNClusters() const { return NClusters_; }
int ShowerAnalyzer::getMaxClustSize() const { return MaxClustSize_; }
float ShowerAnalyzer::getAvgClustSize() const { return AvgClustSize_; }
int ShowerAnalyzer::getPlanesWithClusmore2() const { return PlanesWithClusmore2_; }

float ShowerAnalyzer::getPctHitsFirst10() const { return PctHitsFirst10_; }

float ShowerAnalyzer::getTMin()    const { return tMin_;    }
float ShowerAnalyzer::getTMax()    const { return tMax_;    }
float ShowerAnalyzer::getTMean()   const { return tMean_;   }
float ShowerAnalyzer::getTSpread() const { return tSpread_; }

float ShowerAnalyzer::getNmax() const { return Nmax_; }
float ShowerAnalyzer::getZ0Fit() const { return z0_fit_; }
float ShowerAnalyzer::getXmax() const { return Xmax_; }
float ShowerAnalyzer::getLambda() const { return lambda_; }

float ShowerAnalyzer::getEccentricity3D() const { return eccentricity3D_; }

int ShowerAnalyzer::getNTrackSegments() const { return nTrackSegments_; }

