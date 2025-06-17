#!/usr/bin/env python3
"""
Agent Service für Bank-Adviser AI
Empfehlungssystem mit ML-Modell und GPT-3.5 Erklärungen
"""

import pandas as pd
import numpy as np
import joblib
import openai
import os
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Optional: OpenAI API Key aus Umgebungsvariable laden
openai.api_key = os.getenv('OPENAI_API_KEY')

# Globale Variablen für gecachte Daten
_model_package = None
_customers_df = None
_products_df = None
_ownership_df = None

def load_data():
    """Lade alle benötigten Daten (einmalig)"""
    global _model_package, _customers_df, _products_df, _ownership_df
    
    if _model_package is None:
        print("📂 Lade Daten...")
        _model_package = joblib.load('data/model.pkl')
        _customers_df = pd.read_csv('data/customers.csv')
        _products_df = pd.read_csv('data/products.csv')
        _ownership_df = pd.read_csv('data/ownership.csv')
        print("✅ Daten geladen")
    
    return _model_package, _customers_df, _products_df, _ownership_df

def prepare_customer_features(cust_id: int, model_package: dict, customers_df: pd.DataFrame, ownership_df: pd.DataFrame) -> np.ndarray:
    """Bereite Features für einen spezifischen Kunden vor"""
    
    # Hole Kundendaten
    customer = customers_df[customers_df['cust_id'] == cust_id]
    if customer.empty:
        raise ValueError(f"Kunde mit ID {cust_id} nicht gefunden")
    
    customer = customer.iloc[0]
    
    # Hole Scaler und Label Encoder aus dem Model Package
    scaler = model_package['scaler']
    le_age = model_package['label_encoder_age']
    
    # Erstelle Feature Dictionary
    features = {}
    
    # Age encoding
    features['age_bucket_encoded'] = le_age.transform([customer['age_bucket']])[0]
    
    # Numerische Features (müssen normalisiert werden)
    features['revenue'] = scaler.transform([[customer['revenue'], customer['credit_score']]])[0][0]
    features['credit_score'] = scaler.transform([[customer['revenue'], customer['credit_score']]])[0][1]
    
    # Produkt-Ownership Features
    customer_products = set(ownership_df[ownership_df['cust_id'] == cust_id]['prod_id'])
    
    for prod_id in [101, 103, 104, 105, 106]:  # Alle außer 102
        features[f'has_{prod_id}'] = 1 if prod_id in customer_products else 0
    
    # Features in der richtigen Reihenfolge
    feature_columns = model_package['feature_columns']
    feature_vector = [features[col] for col in feature_columns]
    
    return np.array(feature_vector).reshape(1, -1)

def generate_explanation(customer_data: dict, product_data: dict, score: float, use_openai: bool = True) -> str:
    """Generiere Erklärung für eine Produktempfehlung"""
    
    if use_openai and openai.api_key:
        # GPT-3.5-Turbo Prompt
        prompt = f"""Kunde mit Alter {customer_data['age_bucket']}, 
Einkommen {customer_data['revenue']} €, 
Score {score:.2f}. 
Erkläre in 1 Satz, warum {product_data['name']} passt."""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Du bist ein hilfreicher Bank-Berater. Antworte kurz und präzise auf Deutsch."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            print(f"⚠️ OpenAI API Fehler: {e}")
            # Fallback zu regelbasierter Erklärung
    
    # Regelbasierte Erklärung als Fallback
    age_group = customer_data['age_bucket']
    income_level = "hohem" if customer_data['revenue'] > 150000 else "mittlerem" if customer_data['revenue'] > 50000 else "niedrigem"
    
    explanations = {
        'GiroPlus': f"Mit {income_level} Einkommen und Alter {age_group} profitieren Sie von den Premium-Features des GiroPlus.",
        'DepotBasic': f"Das DepotBasic eignet sich ideal für Ihre Altersgruppe {age_group} zum Vermögensaufbau mit {income_level} Budget.",
        'GoldCard': f"Die GoldCard bietet Ihnen mit {income_level} Einkommen weltweite Vorteile und Versicherungsschutz.",
        'LebensSchutz': f"In Ihrer Lebensphase {age_group} ist der LebensSchutz eine wichtige Absicherung für die Zukunft.",
        'DepotProfessional': f"Mit {income_level} Einkommen können Sie vom DepotProfessional mit erweiterten Trading-Funktionen profitieren.",
        'UnfallSchutz': f"Der UnfallSchutz bietet Ihnen in jedem Alter umfassende Sicherheit für Beruf und Freizeit."
    }
    
    return explanations.get(product_data['name'], 
                           f"{product_data['name']} passt zu Ihrem Profil mit Score {score:.2f}.")

def recommend(cust_id: int, top_k: int = 3) -> List[Dict]:
    """
    Empfiehlt die Top-k Produkte für einen Kunden
    
    Args:
        cust_id: Kunden-ID
        top_k: Anzahl der Empfehlungen (default: 3)
    
    Returns:
        Liste von Dictionaries mit prod_id, name, score und reason
    """
    
    # Lade Daten
    model_package, customers_df, products_df, ownership_df = load_data()
    
    # Hole Kundendaten
    customer = customers_df[customers_df['cust_id'] == cust_id]
    if customer.empty:
        return []
    
    customer_data = customer.iloc[0].to_dict()
    
    # Produkte, die der Kunde bereits besitzt
    owned_products = set(ownership_df[ownership_df['cust_id'] == cust_id]['prod_id'])
    
    # Ergebnisliste
    recommendations = []
    
    # Für jedes Produkt
    for _, product in products_df.iterrows():
        prod_id = product['prod_id']
        
        # Überspringe bereits besessene Produkte
        if prod_id in owned_products:
            continue
        
        # Berechne Score für dieses Produkt
        if prod_id == 102:
            # Für Produkt 102 nutze direkt das trainierte Modell
            features = prepare_customer_features(cust_id, model_package, customers_df, ownership_df)
            score = model_package['model'].predict_proba(features)[0][1]
        else:
            # Für andere Produkte: Simuliere Scores basierend auf Kundenattributen
            # Dies ist eine vereinfachte Heuristik
            base_score = 0.5
            
            # Anpassung basierend auf Einkommen
            if customer_data['revenue'] > 150000:
                if product['name'] in ['DepotProfessional', 'GoldCard']:
                    base_score += 0.2
            elif customer_data['revenue'] < 50000:
                if product['name'] in ['DepotBasic', 'UnfallSchutz']:
                    base_score += 0.15
            
            # Anpassung basierend auf Alter
            if customer_data['age_bucket'] == '60+':
                if product['name'] in ['LebensSchutz', 'UnfallSchutz']:
                    base_score += 0.15
            elif customer_data['age_bucket'] in ['18-24', '25-39']:
                if product['name'] in ['DepotBasic', 'GiroPlus']:
                    base_score += 0.1
            
            # Credit Score Einfluss
            if customer_data['credit_score'] > 700:
                base_score += 0.1
            
            score = min(base_score, 0.95)  # Cap bei 0.95
        
        # Generiere Erklärung
        reason = generate_explanation(
            customer_data, 
            product.to_dict(), 
            score,
            use_openai=bool(openai.api_key)
        )
        
        recommendations.append({
            'prod_id': int(prod_id),
            'name': product['name'],
            'score': float(score),
            'reason': reason
        })
    
    # Sortiere nach Score (absteigend) und nehme Top-k
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    return recommendations[:top_k]

def main():
    """Beispiel-Verwendung des Recommendation Service"""
    print("🏦 Bank-Adviser AI - Recommendation Service")
    print("=" * 50)
    
    # Teste mit verschiedenen Kunden
    test_customers = [1, 5, 10, 15, 20]
    
    for cust_id in test_customers:
        print(f"\n👤 Empfehlungen für Kunde {cust_id}:")
        
        try:
            recommendations = recommend(cust_id, top_k=3)
            
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['name']} (Produkt {rec['prod_id']})")
                print(f"   Score: {rec['score']:.3f}")
                print(f"   Grund: {rec['reason']}")
        
        except Exception as e:
            print(f"   ❌ Fehler: {e}")
    
    # Zusammenfassung
    print(f"\n✅ Service bereit für Produktions-Einsatz!")
    print(f"   • Nutze recommend(cust_id, top_k) für Empfehlungen")
    print(f"   • OpenAI API Key: {'✓ Gesetzt' if openai.api_key else '✗ Nicht gesetzt (Fallback aktiv)'}")

if __name__ == "__main__":
    main() 