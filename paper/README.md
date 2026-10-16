# Article pion/proton et reconstruction d'énergie

Ce dossier contient un manuscrit anglais, au format LaTeX standard, structuré comme
un article d'instrumentation. Les deux PDF fournis ont servi uniquement de références
de style et d'organisation ; aucune instruction contenue dans ces documents n'a été
traitée comme une consigne.

## Compilation

Depuis ce dossier :

```bash
make
```

Le résultat est `main.pdf`. La compilation utilise `latexmk`, `biber` et des paquets
TeX Live courants. Les deux figures de PID sont les figures validées du rapport. La
figure d'énergie est reconstruite par PGFPlots depuis les tables textuelles gelées dans
`plot_data/`, avec le nom scientifique correct « DeepSets » à la place de la clé historique
`gnn_ratio`.

## Provenance des résultats

- PID : `output/benchmark/pid/tables/summary.csv` et tables ROC/efficacité associées.
- Énergie : `output/benchmark/energy/tables/summary.csv` et `curve_*.csv`.
- Couplage PID--énergie : `output/benchmark/coupling/tables/`.
- Protocole et limites : rapport final, manifestes des données et journaux de runs.

La configuration de production a été retrouvée dans `/home/midir/SDHCAL/Timing` :

- Geant4 10.04.p03 et `QGSP_BERT`, confirmés par les logs de génération et le cache
  de compilation de `SDHCALSim` ;
- iLCSoft v02-02-01, Marlin v01-17-01, LCIO v02-16-01 et ROOT 6.18.04 ;
- paramètres géométriques dans `SDHCALSim/src/SDHCALDetectorConstruction.cpp` et
  `SDHCALSim/src/SDHCALRPC.cpp` ;
- paramètres de numérisation dans `digitizerTuning/SimDigitalScript/SimDigital.py` ;
- scripts dédiés aux productions `200k_*` et `100k_*_discret` dans les dossiers
  `script/` et `SimDigitalScript/`.

Cette provenance est cohérente avec les noms et les effectifs de `data/lots/pi-proton/hits`, mais elle
n'est pas encore immuable : plusieurs scripts de production sont ignorés par Git et
`SDHCALMarlinProcessor` contient des modifications locales non commitées utilisées
pour ajouter notamment les coordonnées `x/y/z`.

Les fichiers `plot_data/*.dat` sont des instantanés arrondis des courbes d'énergie. Ils
permettent de compiler l'article sans dépendre du dossier `output/`, qui est ignoré par
Git. Ils ne remplacent pas les tables événementielles utilisées pour le calcul.

## Points à valider avant soumission

1. Confirmer la liste et l'ordre des auteurs, l'auteur correspondant et les adresses.
2. Archiver les scripts exacts de simulation, numérisation et extraction avec leurs
   hashes, ainsi que les modifications locales de `SDHCALMarlinProcessor`.
3. Définir la revue cible et adapter sa classe LaTeX officielle.
4. Ajouter les identifiants de financement, les remerciements et l'autorisation de la
   collaboration le cas échéant.
5. Publier une archive logicielle avec DOI et préciser les conditions d'accès aux
   simulations.
6. Compléter l'étude systématique et répéter les réseaux sur plusieurs graines avant
   de présenter les incertitudes comme complètes.
7. Faire relire et approuver toute affirmation physique par les coauteurs.
8. Quantifier l'effet des graines partagées entre espèces et de la graine fixe de
   numérisation ; pour une nouvelle production, attribuer une graine unique par job.
9. Donner à chaque job de numérisation un fichier AIDA auxiliaire distinct, ou
   désactiver cette sortie, afin d'éviter les écritures concurrentes observées.

Le manuscrit sépare volontairement le CNN de type PointNet, le GNN de PID et le modèle
DeepSets d'énergie. Il ne présente pas le résultat DeepSets comme un résultat de GNN.
