# Rapport PID SDHCAL

Ce dossier contient le rapport de stage M2 en LaTeX sur l'identification hadronique et la reconstruction d'energie dans le SDHCAL.

## Structure

- `main.tex` : fichier principal du rapport.
- `chapters/` : chapitres du manuscrit.
- `Annexes/` : annexes techniques.
- `Fig/` : figures scientifiques du rapport.
- `img/` : logos et images institutionnelles.
- `references.bib` : bibliographie.
- `build/` : fichiers de compilation generes automatiquement.

## Compilation

Depuis ce dossier, lancer :

```sh
make
```

Le PDF final est recopie dans `main.pdf`. Les fichiers temporaires restent dans `build/`.

## Résultats retenus

Cette version constitue le rapport final. Elle retient les performances PID
présentées à DRD6 et les biais recalculés à partir des prédictions événement par
événement. Les courbes historiques de largeur sont interprétées comme des
indicateurs qualitatifs de dispersion et aucun gain numérique de résolution
absolue n'en est déduit. L'étude GNN est présentée comme exploratoire.

## Nettoyage

Pour supprimer les fichiers temporaires :

```sh
make clean
```

Pour supprimer aussi le PDF final :

```sh
make clean-all
```
