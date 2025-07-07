#!/usr/bin/env python3
"""
Einfache Alternative zum Jupyter Notebook
Macht genau das Gleiche - nur als normales Python Script
"""

import pandas as pd

def create_features():
    """Erstellt ML-Features aus den Rohdaten"""
    
    # 1. Daten laden
    customers_df = pd.read_csv('data/customers.csv')
    ownership_df = pd.read_csv('data/ownership.csv')
    
    # 2. One-Hot-Encoding für Produktbesitz
    features_df = customers_df.copy()
    unique_products = ownership_df['prod_id'].unique()
    
    # Stelle sicher, dass alle Produkte von 101-110 vorhanden sind
    all_products = list(range(101, 111))  # 101 bis 110
    
    for prod_id in all_products:
        customers_with_product = ownership_df[ownership_df['prod_id'] == prod_id]['cust_id'].unique()
        features_df[f'has_{prod_id}'] = features_df['cust_id'].isin(customers_with_product).astype(int)
    
    # 3. Features auswählen und speichern
    base_features = ['cust_id', 'age_bucket', 'revenue', 'credit_score']
    product_features = [col for col in features_df.columns if col.startswith('has_')]
    selected_features = base_features + product_features
    
    final_features_df = features_df[selected_features].copy()
    final_features_df.to_parquet('data/features.parquet', index=False)
    
    print(f"✅ Features erstellt: {final_features_df.shape}")
    return final_features_df

if __name__ == "__main__":
    create_features() 