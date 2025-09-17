// count_tracks_ransac.cpp
// C++11 / ROOT: estime le nombre de tracks par évènement via RANSAC 3D
// Hypothèses: TTree "tree" avec branches "x","y","z" (std::vector<float>)
//             Si absent, essaie "I","J","K" (indices pixels) avec pas=10.408 mm, gap=26.131 mm.

#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

struct Vec3 {
    double x, y, z;
};

static inline Vec3 operator-(const Vec3& a, const Vec3& b){ return {a.x-b.x, a.y-b.y, a.z-b.z}; }
static inline double dot(const Vec3& a, const Vec3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline Vec3 cross(const Vec3& a, const Vec3& b){ return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x}; }
static inline double norm(const Vec3& a){ return std::sqrt(dot(a,a)); }
static inline Vec3 normalize(const Vec3& a){ double n = norm(a); return (n>0)? Vec3{a.x/n,a.y/n,a.z/n} : Vec3{0,0,0}; }

// Distance point-droite en 3D (droite passant par P0 avec direction v normalisée)
static inline double pointLineDistance(const Vec3& p, const Vec3& P0, const Vec3& v_unit) {
    // || (p - P0) x v ||
    Vec3 w = p - P0;
    Vec3 c = cross(w, v_unit);
    return norm(c);
}

// Détection multi-tracks par RANSAC
struct RansacParams {
    double tol = 8.0;              // tolérance (mm) pour inliers (ajustez 6–15 mm suivant granularité/bru)
    int    minHitsPerTrack = 12;   // nombre min d'inliers pour valider un track
    int    maxRansacIters = 400;   // itérations RANSAC par piste
    int    maxTracks = 6;          // limite sécurité
    unsigned seed = 12345;
};

struct TrackModel {
    Vec3 P0;    // point sur la droite
    Vec3 v;     // direction UNITAIRE
    std::vector<int> inliers;
};

static TrackModel ransacBestLine(const std::vector<Vec3>& pts,
                                 const std::vector<int>& indices,
                                 const RansacParams& par,
                                 TRandom3& rng)
{
    TrackModel best;
    size_t N = indices.size();
    if (N < 2) return best;

    for (int it = 0; it < par.maxRansacIters; ++it) {
        // échantillonner 2 indices distincts
        int i1 = indices[rng.Integer(N)];
        int i2 = indices[rng.Integer(N)];
        if (i1 == i2) continue;

        const Vec3& P1 = pts[i1];
        const Vec3& P2 = pts[i2];
        Vec3 v = P2 - P1;
        double nv = norm(v);
        if (nv < 1e-6) continue;
        Vec3 v_unit = {v.x/nv, v.y/nv, v.z/nv};

        // compter inliers
        std::vector<int> inliers;
        inliers.reserve(N);
        for (int idx : indices) {
            double d = pointLineDistance(pts[idx], P1, v_unit);
            if (d <= par.tol) inliers.push_back(idx);
        }
        if (inliers.size() > best.inliers.size()) {
            best.P0 = P1;
            best.v  = v_unit;
            best.inliers.swap(inliers);
        }
    }

    // Petit raffinement: refit direction par PCA sur inliers (optionnel mais utile)
    if (best.inliers.size() >= 2) {
        // calcul du barycentre
        Vec3 mu{0,0,0};
        for (int idx : best.inliers) { mu.x += pts[idx].x; mu.y += pts[idx].y; mu.z += pts[idx].z; }
        double invM = 1.0 / best.inliers.size();
        mu.x *= invM; mu.y *= invM; mu.z *= invM;

        // matrice de covariance 3x3 (centrée)
        double Sxx=0,Sxy=0,Sxz=0,Syy=0,Syz=0,Szz=0;
        for (int idx : best.inliers) {
            Vec3 d{pts[idx].x - mu.x, pts[idx].y - mu.y, pts[idx].z - mu.z};
            Sxx += d.x*d.x; Sxy += d.x*d.y; Sxz += d.x*d.z;
            Syy += d.y*d.y; Syz += d.y*d.z; Szz += d.z*d.z;
        }
        // Approx: power iteration pour vecteur propre principal (direction de plus grande variance)
        Vec3 v{1,0,0};
        for (int k = 0; k < 10; ++k) {
            Vec3 w{
                Sxx*v.x + Sxy*v.y + Sxz*v.z,
                Sxy*v.x + Syy*v.y + Syz*v.z,
                Sxz*v.x + Syz*v.y + Szz*v.z
            };
            double n = norm(w);
            if (n < 1e-12) break;
            v = {w.x/n, w.y/n, w.z/n};
        }
        best.P0 = mu;
        best.v  = v;
    }

    return best;
}

// Retire un ensemble d'indices de la "liste active"
static void removeInliers(std::vector<int>& indices, const std::vector<int>& inliers) {
    std::unordered_set<int> S(inliers.begin(), inliers.end());
    indices.erase(std::remove_if(indices.begin(), indices.end(),
                                 [&](int id){ return S.count(id)>0; }),
                  indices.end());
}

static int countTracksEvent(const std::vector<Vec3>& points, const RansacParams& par)
{
    TRandom3 rng(par.seed);
    // indices actifs (points restants)
    std::vector<int> active(points.size());
    std::iota(active.begin(), active.end(), 0);

    int nTracks = 0;
    for (; nTracks < par.maxTracks; ++nTracks) {
        if ((int)active.size() < par.minHitsPerTrack) break;

        TrackModel best = ransacBestLine(points, active, par, rng);
        if ((int)best.inliers.size() < par.minHitsPerTrack) break;

        // (optionnel) critère de longueur le long de z pour éviter les "mini droites"
        double zmin=1e99, zmax=-1e99;
        for (int idx : best.inliers) {
            zmin = std::min(zmin, points[idx].z);
            zmax = std::max(zmax, points[idx].z);
        }
        double spanZ = zmax - zmin;
        if (spanZ < 3.0 * 26.131) { // au moins ~3 couches si gap~26.131 mm
            // pas une vraie piste allongée, on arrête
            break;
        }

        // accepter le track et retirer ses inliers
        removeInliers(active, best.inliers);
    }
    return nTracks;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " 130k_pi_E1to130_digitized.root [treeName=tree]\n";
        return 1;
    }
    std::string inFile = argv[1];
    std::string treeName = (argc >= 3) ? argv[2] : "tree";

    std::unique_ptr<TFile> f(TFile::Open(inFile.c_str(), "READ"));
    if (!f || f->IsZombie()) {
        std::cerr << "Erreur: impossible d'ouvrir " << inFile << "\n";
        return 1;
    }

    TTree* t = dynamic_cast<TTree*>(f->Get(treeName.c_str()));
    if (!t) {
        std::cerr << "Erreur: TTree '" << treeName << "' introuvable.\n";
        return 1;
    }

    // Branches possibles
    std::vector<float>* x = nullptr;
    std::vector<float>* y = nullptr;
    std::vector<float>* z = nullptr;

    std::vector<int>* I = nullptr;
    std::vector<int>* J = nullptr;
    std::vector<int>* K = nullptr;

    bool has_xyz = (t->GetBranch("x") && t->GetBranch("y") && t->GetBranch("z"));
    bool has_IJK = (t->GetBranch("I") && t->GetBranch("J") && t->GetBranch("K"));

    if (has_xyz) {
        t->SetBranchAddress("x", &x);
        t->SetBranchAddress("y", &y);
        t->SetBranchAddress("z", &z);
    } else if (has_IJK) {
        t->SetBranchAddress("I", &I);
        t->SetBranchAddress("J", &J);
        t->SetBranchAddress("K", &K);
    } else {
        std::cerr << "Erreur: ni (x,y,z) ni (I,J,K) présents dans l'arbre.\n";
        return 1;
    }

    // Paramètres RANSAC 
    RansacParams par;
    par.tol = 8.0;              // mm (≈ un pixel si ~10.4 mm)
    par.minHitsPerTrack = 12;   // nombre min de hits alignés
    par.maxRansacIters = 400;
    par.maxTracks = 8;

    const double pixel = 10.408;   // mm
    const double gap   = 26.131;   // mm

    Long64_t n = t->GetEntries();
    std::cout << "Events: " << n << "\n";

    // Sortie: eventNumber (si branche), sinon index local
    int* eventNumber = nullptr;
    if (t->GetBranch("eventNumber")) t->SetBranchAddress("eventNumber", &eventNumber);

    for (Long64_t ievt = 0; ievt < n; ++ievt) {
        t->GetEntry(ievt);

        std::vector<Vec3> pts;
        if (has_xyz) {
            size_t m = std::min({x->size(), y->size(), z->size()});
            pts.reserve(m);
            for (size_t i = 0; i < m; ++i) pts.push_back({(*x)[i], (*y)[i], (*z)[i]});
        } else {
            size_t m = std::min({I->size(), J->size(), K->size()});
            pts.reserve(m);
            for (size_t i = 0; i < m; ++i) {
                // Convertit indices en mm (mêmes conventions que votre reconstruction)
                double xx = (*I)[i] * pixel;
                double yy = (*J)[i] * pixel;
                double zz = ((*K)[i] + 1) * gap;
                pts.push_back({xx, yy, zz});
            }
        }

        int nTracks = countTracksEvent(pts, par);

        int evId = eventNumber ? *eventNumber : static_cast<int>(ievt);
        std::cout << "event " << evId << " : nTrack = " << nTracks << "\n";
    }

    return 0;
}
