# C++ partagé

Code C++ utilisé par plusieurs étapes de l'analyse.

```text
cpp/
├── include/
│   ├── ShowerAnalyzer.h    la classe documentée ci-dessous
│   └── SdhcalPaths.h       résolution des chemins de données, côté C++
└── src/
    └── ShowerAnalyzer.cpp
```

`SdhcalPaths.h` expose la même convention que `src/sdhcal/paths.py` : les chemins
viennent de `SDHCAL_DATA_DIR` et `SDHCAL_OUT_DIR`, et `repoRoot()` remonte
jusqu'à un fichier marqueur, de sorte qu'un macro ROOT peut être lancé depuis
n'importe quel répertoire.

Vérification de la compilation : `make cpp` depuis la racine du dépôt. Pour
construire l'exécutable principal sans lancer l'analyse :
`make analysis/1_shower_features/computeParams`.

## ShowerAnalyzer – Analyse de gerbes hadroniques

`ShowerAnalyzer` est une classe C++ modulaire permettant de calculer automatiquement de nombreux **observables caractéristiques d’une gerbe hadronique**, à partir de vecteurs d’un événement simulé (ou réel), issus par exemple du calorimètre SDHCAL.

---

## Fonctionnalités

La classe permet de calculer les observables suivants :

| Observable                                  | Description                                                           |
| ------------------------------------------- | --------------------------------------------------------------------- |
| `Zbary`, `Zrms`                             | Barycentre et écart-type longitudinal                                 |
| `Rmean`, `Rrms`, `Rskew`                    | Moments radiaux transverses (moyenne, RMS, asymétrie)                 |
| `Radius`                                    | Écart moyen à l’axe longitudinal estimé (via fit linéaire x(z), y(z)) |
| `Thr1`, `Thr2`, `Thr3`                      | Pourcentage de hits seuil 1, 2, 3                                     |
| `RatioThr23`                                | Rapport (Thr2 + Thr3) / Thr1                                          |
| `Zbary_thr2`, `Zbary_thr3`                  | Barycentre Z pour les hits seuil 2 et 3                               |
| `N2`, `N3`                                  | Nombre de hits seuil 2 et 3                                           |
| `Begin`                                     | Première couche significative (≥4 hits sur 4 couches consécutives)    |
| `Thr3ShowerLength`                          | Longueur de gerbe au seuil 3 (en couches consécutives)                |
| `pctHitsFirst10`                            | Pourcentage de hits dans les 10 premières couches (0 à 9)             |
| `Density`                                   | Densité locale moyenne pondérée (fenêtre 3×3)                         |
| `λ₁`, `λ₂`                                  | Valeurs propres de la matrice de covariance XY                        |
| `nClusters`, `maxClustSize`, `avgClustSize` | Analyse des clusters 4-connexes par plan                              |
| `planesWithClusmore2`                       | Nombre de plans contenant au moins 2 clusters                         |

---

## Utilisation

### 1. Inclure l’en-tête

```cpp
#include "ShowerAnalyzer.h"
```

### 2. Préparer les vecteurs d’un événement

```cpp
std::vector<int> vThr, vK, vI, vJ;
std::vector<float> vx, vy, vz;
// Ces vecteurs peuvent être remplis à partir d’un TTree ROOT ou d’un fichier SLCIO converti
```

### 3. Créer un objet et analyser

```cpp
ShowerAnalyzer analyzer(vThr, vK, vx, vy, vz, vI, vJ);
analyzer.analyze();
```

### 4. Récupérer les observables

```cpp
float zbary = analyzer.getZbary();
float radius = analyzer.getRadius();
int nClusters = analyzer.getNClusters();
float ratio = analyzer.getRatioThr23();
```

---

## Dépendances

* **ROOT** (nécessaire pour `TGraph`, `TF1` dans le calcul de `Radius`)
* **C++17 recommandé**

---

## Organisation dans le dépôt

```
cpp/include/ShowerAnalyzer.h              // En-tête
cpp/src/ShowerAnalyzer.cpp                // Implémentation
analysis/1_shower_features/computeParams.cpp   // Boucle sur les événements
```

## Exemple minimal

```cpp
#include "ShowerAnalyzer.h"

int main() {
    std::vector<int> vThr = {1,2,3,1,2,1};
    std::vector<int> vK   = {0,1,2,3,4,5};
    std::vector<int> vI = {5,6,7,8,9,10};
    std::vector<int> vJ = {5,5,5,5,5,5};
    std::vector<float> vx = {1,1,1,1,1,1};
    std::vector<float> vy = {2,2,2,2,2,2};
    std::vector<float> vz = {0,1,2,3,4,5};

    ShowerAnalyzer analyzer(vThr, vK, vx, vy, vz, vI, vJ);
    analyzer.analyze();

    std::cout << "Zbary = " << analyzer.getZbary() << std::endl;
    std::cout << "Rmean = " << analyzer.getRmean() << std::endl;
}
```
