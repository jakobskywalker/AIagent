#!/usr/bin/env python3
"""
Multi-Product ML Model Training für Bank-Adviser AI
Trainiert für jedes Produkt (101-106) ein eigenes LogisticRegression-Modell
und speichert alle Modelle sowie Preprocessing-Objekte in data/model.pkl.
"""

import warnings
warnings.filterwarnings('ignore')

import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from pathlib import Path

# --- Feature Loading ---------------------------------------------------------

def load_features(parquet_path: str = 'data/features.parquet') -> pd.DataFrame:
    """Lädt das Feature-Parquet; bricht mit verständlicher Meldung ab, wenn es fehlt."""
    p = Path(parquet_path)
    if not p.exists():
        raise FileNotFoundError(f"{parquet_path} nicht gefunden – bitte feature Engineering ausführen.")
    return pd.read_parquet(parquet_path)


def prepare_preprocessing(df: pd.DataFrame):
    """Erstellt globalen Scaler und LabelEncoder, gibt transformierte Features zurück."""
    le_age = LabelEncoder().fit(df['age_bucket'])
    df_enc = df.copy()
    df_enc['age_bucket_encoded'] = le_age.transform(df['age_bucket'])

    scaler = StandardScaler().fit(df_enc[['revenue', 'credit_score']])
    df_enc[['revenue', 'credit_score']] = scaler.transform(df_enc[['revenue', 'credit_score']])
    return df_enc, scaler, le_age


def train_models(df_enc: pd.DataFrame, feature_cols: list[int]):
    """Trainiert je Produkt ein LR-Modell und liefert Dicts mit Modellen & AUC-Scores."""
    models = {}
    metrics = {}
    feature_map = {}
    for pid in [101, 102, 103, 104, 105, 106]:
        y = df_enc[f'has_{pid}']
        feature_cols_pid = [c for c in feature_cols if c != f'has_{pid}']
        X = df_enc[feature_cols_pid]
        feature_map[pid] = feature_cols_pid

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        clf = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=1)
        clf.fit(X_train, y_train)

        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        models[pid] = clf
        metrics[pid] = float(auc)
        print(f"Produkt {pid}: ROC-AUC {auc:.3f}")
    return models, metrics, feature_map


def main():
    print("🤖 Multi-Produkt-Training – Bank Adviser AI")
    print("=" * 60)

    df = load_features()
    print(f"Features geladen: {df.shape}")

    df_enc, scaler, le_age = prepare_preprocessing(df)

    feature_cols = [
        'age_bucket_encoded', 'revenue', 'credit_score',
        'has_101', 'has_102', 'has_103', 'has_104', 'has_105', 'has_106'
    ]

    models, metrics, feature_map = train_models(df_enc, feature_cols)

    model_package = {
        'models': models,
        'scaler': scaler,
        'label_encoder_age': le_age,
        'feature_columns_map': feature_map,
        'metrics': metrics
    }

    Path('data').mkdir(exist_ok=True)
    joblib.dump(model_package, 'data/model.pkl')
    print("✅ Modellpaket gespeichert in data/model.pkl")

    print("\nZusammenfassung AUC-Scores:")
    for pid, auc in metrics.items():
        print(f"  • {pid}: {auc:.3f}")


if __name__ == '__main__':
    main() 