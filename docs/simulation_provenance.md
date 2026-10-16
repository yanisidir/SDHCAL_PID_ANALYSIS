# Provenance des simulations

Ce que l'audit signalait comme absent : la configuration Geant4 des lots que
cette analyse exploite. Elle n'est enregistrée ni dans les fichiers ROOT, ni dans
le manifeste des fichiers consolidés. Elle est relevée ici depuis la chaîne de
production, `/home/midir/SDHCAL/Timing`, qui n'est pas versionnée avec ce dépôt.

**À vérifier avant toute publication** : ce document a été établi en lisant les
scripts de production, non en interrogeant les fichiers eux-mêmes, qui ne portent
aucune métadonnée. Si plusieurs campagnes ont utilisé les mêmes noms de sortie,
rien ici ne permet de les distinguer.

## Chaîne de production

| Étape | Outil | Sortie |
| --- | --- | --- |
| Simulation | `SDHCALSim` (Geant4) | `.slcio` |
| Digitalisation | `digitizerTuning/SimDigitalScript` | `.slcio` |
| Conversion | `SDHCALMarlinProcessor/LcioToRootProcessor.py` | `.root` |

Environnement : iLCSoft v02-02-01 sous CVMFS, **Geant4 10.04.p03**
(`geant4-10-04-patch-03`, 8 février 2019).

La version se lit dans l'en-tête des journaux de génération
(`SDHCALSim/script/output*.log`) et dans le cache CMake de la compilation, qui
pointe vers `v02-02-01/geant4/10.04.p03`. Elle ne se lit **pas** dans
`init_ilcsoft.sh`, dont la première ligne nomme une pile plus ancienne
(`v02-00-02/geant4/10.03.p02`) qui n'est pas celle utilisée : c'est la seconde
ligne, `v02-02-01`, que le Readme du projet demande de sourcer.

## Configuration commune aux deux lots

| Paramètre | Valeur | Source |
| --- | --- | --- |
| Liste de physique | **`QGSP_BERT`** | scripts de génération |
| Neutrons | **conservés** (`killNeutrons = False`) | idem |
| Position du faisceau | $z = -20$, gaussienne, $\sigma = 10{,}0$ | idem |
| Direction | $\theta = \phi = 0$, $\sigma_p = 0{,}10$ | idem |
| Coupure temporelle | **1000 ns** (défaut du digitiseur : 600) | `SimDigital.py:189` |
| Temps du hit | `GetPostStepPoint()->GetGlobalTime()` | `SDHCALSteppingAction.cpp:38` |

Les deux lots partagent cette configuration. C'est la condition pour que leur
indépendance soit une force de la méthode plutôt qu'un biais : ils ne diffèrent
que par la graine et par la distribution en énergie.

## Les deux lots

| | Entraînement et calibration | Évaluation |
| --- | --- | --- |
| Script | `single_particle_multiprocessing_continuous_*.py` | `single_particle_multiprocessing_discrete_*.py` |
| Événements | 200 000 par espèce | 5 000 par énergie, soit 100 000 |
| Énergie | uniforme 1–130 GeV | 20 points fixes, 5 à 100 GeV par pas de 5 |
| Fichiers | 40 (un par cœur) | 40 (2 par énergie) |
| Graines | `0` à `39` | `e_idx * 1000 + f` |

Les graines `0` et `1` apparaissent dans les deux lots. Sans conséquence — une
même graine consommée par une simulation différente produit des événements
différents — mais la garantie d'indépendance repose alors sur le raisonnement
plutôt que sur la construction. Un décalage sur l'un des deux lots l'établirait
sans discussion.

## Ce que `QGSP_BERT` implique

Trois points, par ordre d'importance pour cette analyse.

**Il n'y a pas de suffixe `_HP`.** Le transport des neutrons sous 20 MeV utilise
donc les modèles par défaut et non les bibliothèques de sections efficaces
évaluées. Or la composante retardée des gerbes hadroniques — celle qui peuple la
queue temporelle au-delà de quelques nanosecondes — est dominée par ces neutrons
lents. Toute étude portant sur le temps doit traiter cette queue comme
**dépendante du modèle** : elle existe, son ampleur et sa chronologie sont
incertaines.

**`QGSP_BERT` n'est plus la recommandation par défaut.** `FTFP_BERT` l'a remplacée
comme liste de référence pour la calorimétrie hadronique depuis Geant4 9.6, et
c'est d'ailleurs le défaut codé dans `SDHCALSim.cxx` que les scripts de production
surchargent explicitement. Les deux listes diffèrent par le modèle de haute
énergie — chaînes de quarks contre Fritiof — et donnent des multiplicités et des
profils longitudinaux différents.

**L'étude CALICE citée en référence expérimentale utilise `FTFP-BERT-HP`.** La
comparaison du chapitre~5 entre nos résolutions et les données de faisceau
confronte donc deux modélisations différentes, en plus de confronter simulation
et mesure. C'est une raison de plus de la traiter comme un repère et non comme une
validation.

## Ce qui reste non enregistré

Les fichiers de sortie ne portent aucune métadonnée de production. Reconstituer
la configuration d'un fichier donné suppose de retrouver le script qui l'a écrit,
par son nom de sortie. Écrire la liste de physique, la graine et la version de
Geant4 dans une branche ou un fichier annexe au moment de la génération
supprimerait cette dépendance.
