import pandas as pd
import matplotlib.pyplot as plt
from pandas.api import types as ptypes

def analyze_csv(filepath, sort_by=None):
    # Chargement du fichier
    df = pd.read_csv(filepath, parse_dates=True, infer_datetime_format=True)
    
    # 0. Forme et index
    print("=== DataFrame Shape ===")
    print(f"Lignes : {df.shape[0]}, Colonnes : {df.shape[1]}")
    print(f"Type d'index : {type(df.index)}\n")
    
    # 1. Colonnes et types pandas
    print("=== Colonnes et types pandas ===")
    print(df.dtypes, "\n")
    
    # 2. Classification des types de colonnes
    def classify(col):
        if ptypes.is_bool_dtype(df[col]):
            return "Booléen"
        if ptypes.is_numeric_dtype(df[col]):
            return "Numérique"
        if ptypes.is_datetime64_any_dtype(df[col]):
            return "Datetime"
        if ptypes.is_categorical_dtype(df[col]) or ptypes.is_object_dtype(df[col]):
            return "Catégoriel"
        return "Autre"
    
    types_summary = {col: classify(col) for col in df.columns}
    print("=== Classification personnalisée ===")
    for col, typ in types_summary.items():
        print(f"{col:<20} → {typ}")
    print()
    
    # 3. Aperçu des dates si colonnes datetime
    datetime_cols = [c for c, t in types_summary.items() if t == "Datetime"]
    for col in datetime_cols:
        print(f"Colonne '{col}' : min={df[col].min()}, max={df[col].max()}")
    if datetime_cols:
        print()
    
    # 4. Aperçu des données (head & tail)
    print("=== Premieres et dernières lignes ===")
    print(df.head(), "\n", df.tail(), "\n")
    
    # 5. Info détaillée (non-null + mémoire)
    print("=== Info du DataFrame ===")
    df.info()
    print()
    
    # 6. Valeurs manquantes et uniques
    print("=== Valeurs manquantes par colonne ===")
    print(df.isnull().sum(), "\n")
    print("=== Valeurs uniques par colonne ===")
    print(df.nunique(), "\n")
    
    # 7. Option : vérification de l'ordre sur une colonne clé
    if sort_by and sort_by in df.columns:
        is_sorted = df[sort_by].is_monotonic_increasing
        print(f"Tri croissant sur '{sort_by}' : {is_sorted}\n")
    
    # (… puis visualisations et corrélations comme avant …)

if __name__ == "__main__":

    analyze_csv("1k_pi_E1to100_flat.csv", sort_by="Date")  # remplacez "Date" par colonne clé si vous voulez tester le tri
