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
absolue n'en est déduit.

Pour la reconstruction d'énergie, le chiffre retenu est le terme stochastique
$a = 46{,}3 \pm 0{,}6~\%$ du modèle sur hits, moyenne de quatre graines et non
plus d'un unique entraînement : l'écart entre graines (0,0020 sur $\sigma/E$)
est du même ordre que les différences discutées, et le rapport le cite partout
où une comparaison est faite. Le modèle est un encodeur d'ensembles
**DeepSets** — sans arêtes — et non un GNN ; le rapport ne présente aucun
résultat de graphe sur l'énergie. L'apport du temps est donné comme une
structure en énergie (gain sous 25 GeV, rien au-dessus) avec la mise en garde
sur la dégénérescence $a$–$b$ et sur la transformation quantile du temps.

## Nettoyage

Pour supprimer les fichiers temporaires :

```sh
make clean
```

Pour supprimer aussi le PDF final :

```sh
make clean-all
```
