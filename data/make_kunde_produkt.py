#!/usr/bin/env python3
"""
Script to generate customer-product ownership relationships.
Loads customers.csv and products.csv, assigns 1-3 random products to each customer.
Includes individual interest rates for credit products.
"""

import pandas as pd
import random
from datetime import datetime, timedelta

def generate_random_date():
    """Generate a random date within the last 5 years"""
    today = datetime.now()
    five_years_ago = today - timedelta(days=5*365)
    
    # Random number of days between five years ago and today
    random_days = random.randint(0, (today - five_years_ago).days)
    random_date = five_years_ago + timedelta(days=random_days)
    
    # Return as date string in YYYY-MM-DD format
    return random_date.strftime('%Y-%m-%d')

def calculate_individual_interest_rate(base_rate, credit_score, revenue, risk_class):
    """
    Berechnet individuelle Zinssätze basierend auf Kundeneigenschaften
    
    Args:
        base_rate: Basis-Zinssatz des Produkts
        credit_score: Kredit-Score des Kunden (300-850)
        revenue: Jahreseinkommen des Kunden
        risk_class: Risikoklasse des Produkts
    
    Returns:
        Individueller Zinssatz
    """
    # Basis-Anpassungen basierend auf Credit Score
    if credit_score >= 750:
        score_adjustment = -0.5  # Sehr guter Score = niedrigere Zinsen
    elif credit_score >= 650:
        score_adjustment = 0.0
    elif credit_score >= 550:
        score_adjustment = 0.5
    else:
        score_adjustment = 1.0  # Schlechter Score = höhere Zinsen
    
    # Einkommens-Anpassungen
    if revenue >= 100000:
        income_adjustment = -0.3
    elif revenue >= 60000:
        income_adjustment = 0.0
    elif revenue >= 40000:
        income_adjustment = 0.2
    else:
        income_adjustment = 0.5
    
    # Zufällige Variation (+/- 0.25%)
    random_adjustment = random.uniform(-0.25, 0.25)
    
    # Gesamter individueller Zinssatz
    individual_rate = base_rate + score_adjustment + income_adjustment + random_adjustment
    
    # Minimum 1%, Maximum 9%
    return round(max(1.0, min(9.0, individual_rate)), 2)

def generate_ownership():
    """Generate customer-product ownership relationships"""
    
    # Load existing data
    customers_df = pd.read_csv('customers.csv')
    products_df = pd.read_csv('products.csv')
    
    ownership_records = []
    
    # Get all available product IDs
    available_products = products_df['prod_id'].tolist()
    
    # Kreditprodukt-IDs für spezielle Behandlung
    credit_product_ids = [107, 108, 109, 110]
    
    # For each customer, assign 1-3 random products
    for _, customer in customers_df.iterrows():
        customer_id = customer['cust_id']
        credit_score = customer['credit_score']
        revenue = customer['revenue']
        
        # Randomly choose how many products this customer will have (1-3)
        num_products = random.randint(1, 3)
        
        # Randomly select products without duplicates
        selected_products = random.sample(available_products, num_products)
        
        # Create ownership records for this customer
        for product_id in selected_products:
            # Hole Produktdetails
            product = products_df[products_df['prod_id'] == product_id].iloc[0]
            
            ownership_record = {
                'cust_id': customer_id,
                'prod_id': product_id,
                'since_date': generate_random_date()
            }
            
            # Für Kreditprodukte: Füge zusätzliche Felder hinzu
            if product_id in credit_product_ids:
                # Deal-Volumen (Kreditsumme) basierend auf Einkommen
                if revenue >= 100000:
                    deal_volume = random.randint(300000, 800000)
                elif revenue >= 60000:
                    deal_volume = random.randint(200000, 500000)
                else:
                    deal_volume = random.randint(100000, 300000)
                
                ownership_record['deal_volume'] = deal_volume
                
                # Kredittyp
                credit_types = {
                    107: 'Annuitätendarlehen',
                    108: 'Variables Darlehen',
                    109: 'Endfälliges Darlehen',
                    110: 'KfW-Darlehen'
                }
                ownership_record['credit_type'] = credit_types.get(product_id, 'Standard')
                
                # Individuelle Zinsen berechnen
                base_rate = product['interest_rate'] if pd.notna(product['interest_rate']) else 4.0
                individual_rate = calculate_individual_interest_rate(
                    base_rate, credit_score, revenue, product['risk_class']
                )
                ownership_record['interest_rate'] = individual_rate
                
                # Individuelle Risikoklasse basierend auf Kundenmerkmalen
                if credit_score >= 700 and revenue >= 80000:
                    ownership_record['risk_class'] = 'niedrig'
                elif credit_score >= 600 and revenue >= 50000:
                    ownership_record['risk_class'] = 'mittel'
                else:
                    ownership_record['risk_class'] = 'hoch'
            else:
                # Für Nicht-Kredit-Produkte: NULL-Werte
                ownership_record['deal_volume'] = None
                ownership_record['credit_type'] = None
                ownership_record['interest_rate'] = None
                ownership_record['risk_class'] = None
            
            ownership_records.append(ownership_record)
    
    return ownership_records

def main():
    """Main function to generate and save ownership data"""
    
    # Generate ownership data
    ownership_data = generate_ownership()
    
    # Create DataFrame
    df = pd.DataFrame(ownership_data)
    
    # Sort by customer ID and product ID for better readability
    df = df.sort_values(['cust_id', 'prod_id']).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv('ownership.csv', index=False)
    
    # Success message with exact format requested
    print(f'ownership.csv erstellt, {len(df)} Zeilen')
    
    # Optional: Show some statistics
    print(f"\n📊 Übersicht:")
    print(f"   • Durchschnittliche Produkte pro Kunde: {len(df) / df['cust_id'].nunique():.1f}")
    print(f"   • Produktverteilung:")
    product_counts = df['prod_id'].value_counts().sort_index()
    for prod_id, count in product_counts.items():
        print(f"     - Produkt {prod_id}: {count} Kunden")
    
    # Zeige Kredit-Statistiken
    credit_df = df[df['interest_rate'].notna()]
    if not credit_df.empty:
        print(f"\n💰 Kredit-Statistiken:")
        print(f"   • Anzahl Kredite: {len(credit_df)}")
        print(f"   • Durchschnittlicher Zinssatz: {credit_df['interest_rate'].mean():.2f}%")
        print(f"   • Zinssatz-Range: {credit_df['interest_rate'].min():.2f}% - {credit_df['interest_rate'].max():.2f}%")
        print(f"   • Durchschnittliches Deal-Volumen: €{credit_df['deal_volume'].mean():,.0f}")
        
        print(f"\n   • Risikoklassen-Verteilung:")
        risk_dist = credit_df['risk_class'].value_counts()
        for risk, count in risk_dist.items():
            print(f"     - {risk}: {count} Kredite")

if __name__ == '__main__':
    main() 