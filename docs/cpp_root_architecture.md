# Architecture C++ et interfaces ROOT

## Chaîne d'extraction

La chaîne R9 conserve le flux scientifique historique :

```text
TFile/TTree d'entrée
  -> sdhcal::bindShowerInput
  -> vecteurs thr, K, x, y, z, I, J, time
  -> sdhcal::extractShowerFeatures
  -> ShowerAnalyzer::analyze
  -> sdhcal::ShowerFeatures
  -> branches ROOT de sortie
```

`cpp/include/ShowerAnalyzer.h` et `cpp/src/ShowerAnalyzer.cpp` restent la source
des calculs scientifiques. R9 ne change ni leur ordre d'appel, ni leurs formules,
ni leurs tolérances, ni les conventions d'échec des fits. La classe dépend encore
de ROOT pour les fits, histogrammes et valeurs propres ; l'extraction R9 n'essaie
pas de réécrire ces algorithmes.

`cpp/include/ShowerFeatures.h` définit l'enregistrement plat écrit dans ROOT.
`cpp/src/ShowerFeatures.cpp` construit exactement un `ShowerAnalyzer`, appelle
`analyze()`, copie ses getters, calcule les deux agrégats historiques
`nHitsTotal` et `sumThrTotal`, et expose le drapeau `K == 47`.

`cpp/include/RootShowerIO.h` et `cpp/src/RootShowerIO.cpp` portent uniquement le
contrat ROOT partagé : adresses des branches d'entrée, schéma de `paramsTree`,
schéma des branches ajoutées et remplissage des seules branches ajoutées. Cette
couche ne définit aucun observable de gerbe.

Les fichiers `analysis/1_shower_features/extract.cpp` et
`extract_parallel.cpp` restent les points d'entrée historiques. Ils ouvrent les
fichiers, choisissent l'arbre, orchestrent la boucle d'événements et écrivent la
sortie. Leurs arguments positionnels et codes de sortie CLI sont conservés.

## Contrat ROOT

L'entrée principale est l'arbre demandé sur la ligne de commande, généralement
`tree`. Les branches obligatoires historiquement liées sont :

| Branche | Type ROOT attendu |
|---|---|
| `particle1PDG` | `Int_t` |
| `thr`, `K`, `I`, `J` | `std::vector<int>` |
| `x`, `y`, `z`, `time` | `std::vector<float>` |

`eventNumber` (`Int_t`) est optionnel. En son absence, l'indice de ligne est
écrit avec l'avertissement historique. `energy` (`Float_t`) est optionnel pour
`extract` ; lorsqu'il existe, il devient `primaryEnergy` (`Float_t`).

`extract` écrit un arbre `paramsTree`, une entrée par événement et dans le même
ordre. Son schéma ordonné est :

```text
eventNumber, [primaryEnergy], particlePDG,
Thr1, Thr2, Thr3, Begin, Radius, Density, NClusters, ratioThr23,
Zbary, Zrms, PctHitsFirst10, PlanesWithClusmore2,
AvgClustSize, MaxClustSize, lambda1, lambda2, N3, N2, N1,
tMin, tMax, tMean, tSpread, Nmax, z0_fit, Xmax, lambda,
nTrackSegments, eccentricity3D, nHitsTotal, sumThrTotal,
touchesLastLayer
```

Les identifiants, compteurs et drapeaux utilisent `/I`; les observables et
énergies utilisent `/F`. `extract_parallel` conserve le nom de l'arbre source,
clone toutes ses branches et ajoute les observables sans changer l'ordre des
événements ou des hits.

Les consommateurs Python restent fondés sur `paramsTree`, les noms ci-dessus et
les couples `(particlePDG, eventNumber)`. `src/sdhcal/rootio.py` lit les arbres
consolidés ; `src/sdhcal/shards.py` accepte `tree` ou `paramsTree`, convertit les
identifiants locaux de shards et dérive `touchesLastLayer` pour les anciens lots.

## Construction et exécution

Les commandes historiques restent valides :

```bash
make features
make shower-analyzer
./analysis/1_shower_features/extract input.root tree output.root
./analysis/1_shower_features/extract_parallel input.root output.root tree
IN=... OUT=... analysis/1_shower_features/parallel.sh
```

`ROOT_CONFIG` permet de sélectionner un environnement ROOT non présent dans le
`PATH`, par exemple :

```bash
make cpp ROOT_CONFIG=/chemin/vers/root-config
make cpp-test ROOT_CONFIG=/chemin/vers/root-config
```

Le Makefile conserve C++17, les avertissements et l'optimisation historiques,
puis ajoute les drapeaux et bibliothèques retournés par `root-config`.

## Tests et référence numérique

`tests/cpp/test_shower_features.cpp` caractérise des événements vide, mono-hit
et multi-hit. Les comparaisons déterministes sont exactes ; les observables à
virgule flottante utilisent `1e-6` à `1e-5`, et les paramètres du minimiseur
Gaisser–Hillas `1e-3`, afin de couvrir les variations du minimiseur ROOT sans
masquer une dérive en simple précision.

`tests/test_cpp_root_pipeline.py` compile les exécutables dans un dossier
temporaire, produit de petits TTrees, vérifie CLI, arbres, branches, types,
ordre, valeurs et fallback d'identifiant. L'inspection utilise PyROOT provenant
du même environnement que `root-config`.

## Défauts techniques connus

`extract_parallel` clone la branche `eventNumber` puis en crée une seconde du
même nom. Le défaut apparaît dès que l'entrée contient `eventNumber` ; une
seconde exécution du programme duplique également toutes les branches dérivées.
Les tests correspondants sont `xfail`. R9 préserve le comportement et ne choisit
pas silencieusement entre rejet, réutilisation et remplacement. La correction
doit être une modification isolée avec une politique annoncée pour les anciens
fichiers.

Les branches obligatoires et la cohérence des tailles des vecteurs ne sont pas
validées avant leur déréférencement. Les cas branche absente et type incorrect
sont également documentés en `xfail`; leur sécurisation reste une correction
technique séparée.

Dans l'environnement R9, PyROOT 6.40.04 relit correctement les fichiers, mais
`uproot 5.7.7` se bloque sur les paniers produits par cette version de ROOT. Le
test de round-trip Python est donc explicitement ignoré après timeout. Le schéma
requis par les consommateurs Python est vérifié, mais ce round-trip devra être
rejoué avec une combinaison ROOT/uproot compatible avant de déclarer cette
partie totalement validée.

Ces limites sont techniques. Elles ne valident ni ne corrigent les définitions
physiques, les sélections, les calibrations ou les conventions numériques.
