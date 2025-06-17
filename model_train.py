#!/usr/bin/env python3
"""
Machine Learning Model Training für Bank-Adviser AI
Trainiert ein LogisticRegression Modell zur Vorhersage von Produkt 102 (DepotBasic)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

def prepare_features(df):
    """Prepare features for machine learning"""
    
    # Erstelle eine Kopie des DataFrames
    features_df = df.copy()
    
    # Encode age_bucket kategorisch
    le_age = LabelEncoder()
    features_df['age_bucket_encoded'] = le_age.fit_transform(features_df['age_bucket'])
    
    # Normalisiere numerische Features
    scaler = StandardScaler()
    numerical_features = ['revenue', 'credit_score']
    features_df[numerical_features] = scaler.fit_transform(features_df[numerical_features])
    
    # Wähle Features für das Modell
    feature_columns = [
        'age_bucket_encoded', 'revenue', 'credit_score',
        'has_101', 'has_103', 'has_104', 'has_105', 'has_106'  # Alle anderen Produkte außer 102
    ]
    
    X = features_df[feature_columns]
    
    return X, scaler, le_age, feature_columns

def main():
    print("🤖 ML Model Training - Bank Adviser AI")
    print("=" * 50)
    
    # Lade Features
    print("\n📂 Lade features.parquet...")
    try:
        df = pd.read_parquet('data/features.parquet')
        print(f"Features geladen: {df.shape}")
        print(f"Spalten: {df.columns.tolist()}")
    except FileNotFoundError:
        print("❌ Fehler: features.parquet nicht gefunden!")
        print("Bitte führe zuerst das Feature Engineering aus.")
        return
    
    # Zielvariable definieren
    print(f"\n🎯 Zielvariable: has_102 (Produkt 102 - DepotBasic)")
    y = df['has_102']
    
    # Zeige Klassenverteilung
    class_distribution = y.value_counts()
    print(f"Klassenverteilung:")
    print(f"   • Kein Produkt 102: {class_distribution[0]} Kunden ({class_distribution[0]/len(y)*100:.1f}%)")
    print(f"   • Hat Produkt 102: {class_distribution[1]} Kunden ({class_distribution[1]/len(y)*100:.1f}%)")
    
    # Features vorbereiten
    print(f"\n🔧 Features vorbereiten...")
    X, scaler, le_age, feature_columns = prepare_features(df)
    print(f"Finale Features: {feature_columns}")
    print(f"Feature Matrix: {X.shape}")
    
    # Train/Test Split (80/20)
    print(f"\n📊 Train/Test Split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training Set: {X_train.shape}")
    print(f"Test Set: {X_test.shape}")
    
    # Modell trainieren
    print(f"\n🧠 LogisticRegression Training...")
    model = LogisticRegression(
        random_state=42,
        class_weight='balanced',  # Ausgleich für unbalancierte Klassen
        max_iter=1000
    )
    
    model.fit(X_train, y_train)
    print(f"✅ Modell erfolgreich trainiert!")
    
    # Vorhersagen
    print(f"\n📈 Modell Evaluation...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Metriken berechnen
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_roc_auc = roc_auc_score(y_train, y_train_proba)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    
    # Ergebnisse ausgeben
    print(f"\n🎯 Modell Performance:")
    print(f"   Training Set:")
    print(f"     • Accuracy: {train_accuracy:.4f}")
    print(f"     • ROC-AUC: {train_roc_auc:.4f}")
    print(f"   Test Set:")
    print(f"     • Accuracy: {test_accuracy:.4f}")
    print(f"     • ROC-AUC: {test_roc_auc:.4f}")
    
    # Feature Importance
    print(f"\n🔍 Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    for _, row in feature_importance.iterrows():
        direction = "📈" if row['coefficient'] > 0 else "📉"
        print(f"   {direction} {row['feature']}: {row['coefficient']:.4f}")
    
    # Confusion Matrix
    print(f"\n📊 Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"                 Predicted")
    print(f"                 No   Yes")
    print(f"Actual    No  [{cm[0,0]:3d}   {cm[0,1]:3d}]")
    print(f"          Yes [{cm[1,0]:3d}   {cm[1,1]:3d}]")
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['No Product 102', 'Has Product 102']))
    
    # Modell und Preprocessing-Objekte speichern
    print(f"\n💾 Speichere Modell und Preprocessing...")
    
    # Erstelle ein Dictionary mit allen benötigten Objekten
    model_package = {
        'model': model,
        'scaler': scaler,
        'label_encoder_age': le_age,
        'feature_columns': feature_columns,
        'performance': {
            'test_accuracy': test_accuracy,
            'test_roc_auc': test_roc_auc,
            'train_accuracy': train_accuracy,
            'train_roc_auc': train_roc_auc
        }
    }
    
    joblib.dump(model_package, 'data/model.pkl')
    print(f"✅ Modell gespeichert als 'data/model.pkl'")
    
    # Zusammenfassung
    print(f"\n🎉 Training erfolgreich abgeschlossen!")
    print(f"   • Accuracy: {test_accuracy:.4f}")
    print(f"   • ROC-AUC: {test_roc_auc:.4f}")
    print(f"   • Modell: data/model.pkl")

if __name__ == "__main__":
    main() 