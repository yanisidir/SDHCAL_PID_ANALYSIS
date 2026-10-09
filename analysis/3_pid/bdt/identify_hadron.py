#!/usr/bin/env python3
""" identifiy_hadron.py
Ce script implémente un classifieur supervisé utilisant la méthode AdaBoost (Adaptive Boosting),
qui combine plusieurs arbres de décision peu profonds (appelés estimateurs faibles) pour former un modèle plus robuste.
L'entraînement est optimisé à l'aide d'une recherche en grille (GridSearchCV) afin de déterminer les meilleurs hyperparamètres
tels que le nombre d'estimateurs, le taux d'apprentissage et la profondeur des arbres.
Le modèle final est évalué en termes de précision et d'autres métriques de classification,
et l'importance relative des variables (features) est visualisée pour interpréter les résultats.
"""

import matplotlib.pyplot as plt
import joblib
import pandas as pd
import uproot
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

from sdhcal import paths

# 1) Chargement et étiquetage
def load_data(base_dirs):
    frames = []
    for label, path in base_dirs.items():
        df = uproot.open(path)["paramsTree"].arrays(library="pd")
        df["label"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

# 2) Préparation des features
def prepare_xy(df, features):
    return df[features], df["label"]

# 3) Entraînement avec GridSearch sur AdaBoost
def train_boosted_tree(X_train, y_train):
    # base estimator : arbre peu profond
    base_clf = DecisionTreeClassifier(random_state=42)
    model = AdaBoostClassifier(estimator=base_clf, random_state=42)
    
    param_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1.0],
        # on ajuste la profondeur du tree de base via le préfixe estimator__
        "estimator__max_depth": [1, 3, 5]
    }
    grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

# 4) Évaluation
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# 5) Visualisation des importances et sauvegarde
def plot_importances_and_save(model, features, out_png, out_pkl):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    plt.figure(figsize=(8,6))
    plt.title("Feature importances (AdaBoost)")
    plt.bar(range(len(features)), importances[indices], align="center")
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(out_png)
    joblib.dump(model, out_pkl)
    plt.close()

if __name__ == "__main__":
    base_dirs = {
        "pion": paths.dataset("params", "pi"),
        "proton": paths.dataset("params", "proton"),
        "kaon": paths.dataset("params", "kaon"),
    }
    features = ["Thr3","Begin","Radius","Density","nClusters"]

    # pipeline
    df = load_data(base_dirs)
    X, y = prepare_xy(df, features)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model, best_params = train_boosted_tree(X_train, y_train)
    print("Meilleurs paramètres :", best_params)
    evaluate(model, X_test, y_test)
    model_dir = paths.model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_importances_and_save(
        model, features,
        model_dir / "boosted_tree_importances.png",
        model_dir / "boosted_hadron_tree.pkl"
    )
