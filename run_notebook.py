#!/usr/bin/env python3
"""
Führt das Feature Engineering Notebook als Python Script aus
"""

print("=" * 60)
print("🔬 Feature Engineering für Bank-Adviser AI")
print("=" * 60)

# Zelle 1: Import pandas und laden der CSV-Dateien
print("\n📂 Zelle 1: Lade CSV-Dateien")
print("-" * 40)

import pandas as pd
import numpy as np

# Lade Kundendaten
customers_df = pd.read_csv('data/customers.csv')
print(f"Customers geladen: {len(customers_df)} Zeilen")
print(f"Spalten: {customers_df.columns.tolist()}")

# Lade Ownership-Daten
ownership_df = pd.read_csv('data/ownership.csv')
print(f"\nOwnership geladen: {len(ownership_df)} Zeilen")
print(f"Spalten: {ownership_df.columns.tolist()}")

# Zeige erste Zeilen
print("\nErste 3 Kunden:")
print(customers_df.head(3).to_string(index=False))

# Zelle 2: Erstelle One-Hot-Encoding für Produktbesitz
print("\n\n📊 Zelle 2: One-Hot-Encoding für Produktbesitz")
print("-" * 40)

# Alle verfügbaren Produkt-IDs ermitteln
unique_products = ownership_df['prod_id'].unique()
print(f"Verfügbare Produkte: {sorted(unique_products)}")

# Starte mit Kundendaten
features_df = customers_df.copy()

# Erstelle für jedes Produkt eine has_<prod_id> Spalte
for prod_id in unique_products:
    # Kunden, die dieses Produkt besitzen
    customers_with_product = ownership_df[ownership_df['prod_id'] == prod_id]['cust_id'].unique()
    
    # Erstelle One-Hot-Spalte
    column_name = f'has_{prod_id}'
    features_df[column_name] = features_df['cust_id'].isin(customers_with_product).astype(int)
    
    print(f"Spalte {column_name} erstellt: {features_df[column_name].sum()} Kunden haben dieses Produkt")

print(f"\nFeatures DataFrame: {features_df.shape}")

# Zelle 3: Features auswählen und als Parquet speichern
print("\n\n💾 Zelle 3: Features auswählen und speichern")
print("-" * 40)

# Definiere gewünschte Features
base_features = ['cust_id', 'age_bucket', 'revenue', 'credit_score']
product_features = [col for col in features_df.columns if col.startswith('has_')]

# Alle Features zusammenfassen
selected_features = base_features + product_features
print(f"Gewählte Features: {selected_features}")

# Features DataFrame erstellen
final_features_df = features_df[selected_features].copy()

# Statistiken anzeigen
print(f"\n📊 Feature-Statistiken:")
print(f"   • Anzahl Kunden: {len(final_features_df)}")
print(f"   • Anzahl Features: {len(selected_features)}")
print(f"   • Durchschnittliches Revenue: €{final_features_df['revenue'].mean():,.0f}")
print(f"   • Durchschnittlicher Credit Score: {final_features_df['credit_score'].mean():.0f}")

print(f"\n🎯 Altersverteilung:")
age_dist = final_features_df['age_bucket'].value_counts().sort_index()
for age, count in age_dist.items():
    print(f"   • {age}: {count} Kunden ({count/len(final_features_df)*100:.1f}%)")

print(f"\n📦 Produktbesitz-Verteilung:")
for col in product_features:
    count = final_features_df[col].sum()
    percentage = count / len(final_features_df) * 100
    print(f"   • {col}: {count} Kunden ({percentage:.1f}%)")

# Zeige erste Zeilen
print(f"\n📋 Final Features DataFrame (erste 5 Zeilen):")
print(final_features_df.head().to_string(index=False))

print("\n" + "=" * 60)
print("✅ Feature Engineering abgeschlossen!")
print("=" * 60) 