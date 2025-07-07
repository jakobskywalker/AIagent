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

def build_feature_vector(cust_id: int, feature_columns: list[str], customers_df: pd.DataFrame, ownership_df: pd.DataFrame, scaler, le_age) -> np.ndarray:
    """Erstellt Feature-Vektor für gewünschte Spaltenliste"""
    
    # Hole Kundendaten
    customer = customers_df[customers_df['cust_id'] == cust_id]
    if customer.empty:
        raise ValueError(f"Kunde mit ID {cust_id} nicht gefunden")
    
    customer = customer.iloc[0]
    
    # scaler und le_age werden übergeben
    
    # Erstelle Feature Dictionary
    features = {}
    
    # Age encoding
    features['age_bucket_encoded'] = le_age.transform([customer['age_bucket']])[0]
    
    # Numerische Features (müssen normalisiert werden)
    features['revenue'] = scaler.transform([[customer['revenue'], customer['credit_score']]])[0][0]
    features['credit_score'] = scaler.transform([[customer['revenue'], customer['credit_score']]])[0][1]
    
    # Produkt-Ownership Features
    customer_products = set(ownership_df[ownership_df['cust_id'] == cust_id]['prod_id'])
    
    for prod_id in [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]:
        features[f'has_{prod_id}'] = 1 if prod_id in customer_products else 0
    feature_vector = [features[col] for col in feature_columns]
    
    return np.array(feature_vector).reshape(1, -1)

# Backwards compatibility for older single-model usage
def prepare_customer_features(cust_id: int, model_package: dict, customers_df: pd.DataFrame, ownership_df: pd.DataFrame):
    feature_columns = model_package.get('feature_columns') or list(model_package['feature_columns_map'].values())[0]
    return build_feature_vector(cust_id, feature_columns, customers_df, ownership_df, model_package['scaler'], model_package['label_encoder_age'])

def generate_explanation(customer_data: dict, product_data: dict, score: float, use_openai: bool = True) -> str:
    """Generiere Erklärung für eine Produktempfehlung"""
    
    if use_openai and openai.api_key:
        # GPT-3.5-Turbo Prompt
        prompt = f"""Kunde mit Alter {customer_data['age_bucket']}, 
Einkommen {customer_data['revenue']} €, 
Score {score:.2f}. 
Erkläre in 1 Satz, warum {product_data['name']} passt."""
        
        try:
            client = openai.OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Du bist ein hilfreicher Bank-Berater. Antworte kurz und präzise auf Deutsch."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
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
        'UnfallSchutz': f"Der UnfallSchutz bietet Ihnen in jedem Alter umfassende Sicherheit für Beruf und Freizeit.",
        'BauFinanz Standard': f"Mit {income_level} Einkommen und Alter {age_group} ist die klassische Baufinanzierung ideal für Ihren Immobilienerwerb.",
        'WohnTraum Flex': f"Die flexible Immobilienfinanzierung passt zu Ihrer Lebenssituation mit {income_level} Einkommen und bietet Spielraum.",
        'ImmoInvest Pro': f"Als Kapitalanleger mit {income_level} Einkommen nutzen Sie die Vorteile der ImmoInvest Pro Finanzierung.",
        'ErstHeim Bonus': f"Als Erstkäufer im Alter {age_group} profitieren Sie von staatlicher Förderung und günstigen Konditionen."
    }
    
    return explanations.get(product_data['name'], 
                           f"{product_data['name']} passt zu Ihrem Profil mit Score {score:.2f}.")

def recommend(cust_id: int, top_k: int = 3, scenario: str = "Ganzheitliche Beratung", **kwargs) -> List[Dict]:
    """
    Empfiehlt die Top-k Produkte für einen Kunden basierend auf dem Beratungsanlass
    
    Args:
        cust_id: Kunden-ID
        top_k: Anzahl der Empfehlungen (default: 3)
        scenario: Beratungsanlass
        **kwargs: Zusätzliche Parameter (z.B. financing_need)
    
    Returns:
        Liste von Dictionaries mit prod_id, name, score, reason und is_cross_sell
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
    
    # Definiere Produktkategorien für verschiedene Szenarien
    scenario_products = {
        "Immobilienfinanzierung": {
            'primary': [107, 108, 109, 110],  # Kredite
            'cross_sell': [104, 106, 101]  # Lebensschutz, Unfallschutz, GiroPlus
        },
        "Kontoeröffnung": {
            'primary': [101],  # GiroPlus
            'cross_sell': [103, 102, 104]  # GoldCard, DepotBasic, LebensSchutz
        },
        "Vermögensaufbau": {
            'primary': [102, 105],  # Depots
            'cross_sell': [101, 103]  # GiroPlus, GoldCard
        },
        "Absicherung & Vorsorge": {
            'primary': [104, 106],  # Versicherungen
            'cross_sell': [101]  # GiroPlus
        },
        "Ganzheitliche Beratung": {
            'primary': list(range(101, 111)),  # Alle Produkte
            'cross_sell': []
        }
    }
    
    # Hole relevante Produkte für das Szenario
    relevant_products = scenario_products.get(scenario, scenario_products["Ganzheitliche Beratung"])
    primary_products = relevant_products['primary']
    cross_sell_products = relevant_products['cross_sell']
    
    # Ergebnisliste
    recommendations = []
    
    # Bewerte primäre Produkte
    for _, product in products_df.iterrows():
        prod_id = product['prod_id']
        
        # Skip bereits besessene Produkte
        if prod_id in owned_products:
            continue
        
        # Berechne Score
        if prod_id in model_package['models']:
            # ML-basierter Score
            model = model_package['models'][prod_id]
            if 'feature_map' in model_package:
                feature_columns = model_package['feature_map'][prod_id]
            else:
                feature_columns = model_package['feature_columns_map'][prod_id]
            
            try:
                feature_vector = build_feature_vector(
                    cust_id, feature_columns, customers_df, ownership_df, 
                    model_package['scaler'], model_package['label_encoder_age']
                )
                score = model.predict_proba(feature_vector)[0][1]

                # XAI: Beitrag jedes Features (nur für LR)
                contributions = None
                try:
                    coeffs = model.coef_[0]
                    contrib_raw = coeffs * feature_vector.flatten()
                    contributions = [
                        {
                            'feature': feature_columns[i],
                            'value': float(feature_vector.flatten()[i]),
                            'coefficient': float(coeffs[i]),
                            'contribution': float(contrib_raw[i])
                        }
                        for i in range(len(feature_columns))
                    ]
                    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
                except Exception:
                    contributions = None
            except Exception as e:
                print(f"Fehler bei Vorhersage für Produkt {prod_id}: {e}")
                score = 0.5
                contributions = None
        else:
            score = 0.5
        
        # Anpasse Score basierend auf Szenario
        is_primary = prod_id in primary_products
        is_cross_sell = prod_id in cross_sell_products
        
        if scenario != "Ganzheitliche Beratung":
            if is_primary:
                score *= 2.0  # Priorisiere Hauptprodukte
            elif is_cross_sell:
                score *= 1.3  # Boost für Cross-Sell
            else:
                score *= 0.1  # Deprioritisiere irrelevante Produkte
        
        # Spezielle Anpassungen für Immobilienfinanzierung
        if scenario == "Immobilienfinanzierung" and prod_id in [107, 108, 109, 110]:
            financing_need = kwargs.get('financing_need', 0)
            ltv_ratio = kwargs.get('ltv_ratio', 80)
            
            # Passe Score basierend auf Finanzierungsbedarf an
            if prod_id == 110 and ltv_ratio <= 90:  # ErstHeim Bonus
                score *= 1.5  # Bonus für Erstkäufer-geeignete Beleihung
            elif prod_id == 109 and financing_need > 500000:  # ImmoInvest Pro
                score *= 1.3  # Gut für große Finanzierungen
        
        # Generiere Erklärung
        product_data = product.to_dict()
        reason = generate_explanation(customer_data, product_data, score, use_openai=False)
        
        # Füge Cross-Sell Info hinzu
        if is_cross_sell and scenario != "Ganzheitliche Beratung":
            reason = "🔗 " + reason + " (Sinnvolle Ergänzung)"
        
        recommendations.append({
            'prod_id': prod_id,
            'name': product['name'],
            'score': score,
            'reason': reason,
            'category': product['category'],
            'is_cross_sell': is_cross_sell,
            'is_primary': is_primary,
            'contributions': contributions
        })
    
    # Sortiere und filtere
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    # Stelle sicher, dass mindestens ein Hauptprodukt dabei ist
    if scenario != "Ganzheitliche Beratung":
        primary_recs = [r for r in recommendations if r['is_primary']]
        cross_sell_recs = [r for r in recommendations if r['is_cross_sell']]
        other_recs = [r for r in recommendations if not r['is_primary'] and not r['is_cross_sell']]
        
        # Nimm top primäre Produkte + top Cross-Sell
        num_primary = min(len(primary_recs), max(1, top_k // 2))
        num_cross_sell = min(len(cross_sell_recs), top_k - num_primary)
        
        final_recommendations = primary_recs[:num_primary] + cross_sell_recs[:num_cross_sell]
        
        # Fülle auf, falls nötig
        if len(final_recommendations) < top_k:
            remaining = top_k - len(final_recommendations)
            final_recommendations.extend(other_recs[:remaining])
        
        return final_recommendations[:top_k]
    else:
        return recommendations[:top_k]

def top_potential_analysis(top_k_per_customer: int = 3):
    """Berechnet für alle Kunden den erwarteten Gewinn basierend auf Produktempfehlungen

    Args:
        top_k_per_customer: Wie viele Empfehlungen pro Kunde berücksichtigt werden sollen

    Returns:
        Dict mit Top-Kunden und deren erwartetem Gewinn sowie Detailinformationen
    """
    # Lade Daten (ggf. aus Cache)
    model_package, customers_df, products_df, ownership_df = load_data()

    results = []

    for cust_id in customers_df['cust_id']:
        # Empfehlungen für diesen Kunden holen
        recs = recommend(cust_id, top_k=top_k_per_customer)
        expected_profit = 0.0
        detailed_recs = []

        for rec in recs:
            # Preis des Produkts ermitteln
            price_row = products_df[products_df['prod_id'] == rec['prod_id']]
            if price_row.empty:
                continue
            price = float(price_row.iloc[0]['price'])
            exp_profit = price * rec['score']
            expected_profit += exp_profit

            detailed_recs.append({
                **rec,
                'price': price,
                'expected_profit': exp_profit
            })

        results.append({
            'cust_id': int(cust_id),
            'expected_profit': expected_profit,
            'recommendations': detailed_recs
        })

    # Sortiere nach erwartetem Gewinn absteigend
    results.sort(key=lambda x: x['expected_profit'], reverse=True)

    return results

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