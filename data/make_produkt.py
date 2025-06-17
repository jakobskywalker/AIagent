#!/usr/bin/env python3
"""
Script to generate bank product data.
Creates 6 bank products and saves them as products.csv.
"""

import pandas as pd

def generate_products():
    """Generate 6 bank product data entries"""
    
    products = [
        {
            'prod_id': 101,
            'name': 'GiroPlus',
            'category': 'Giro',
            'risk_class': 'niedrig',
            'price': 89.90,
            'short_desc': 'Premium Girokonto mit kostenlosen Überweisungen und Kreditkarte inklusive'
        },
        {
            'prod_id': 102,
            'name': 'DepotBasic',
            'category': 'Depot',
            'risk_class': 'mittel',
            'price': 12.50,
            'short_desc': 'Günstige Wertpapierdepot für Einsteiger mit reduzierten Ordergebühren'
        },
        {
            'prod_id': 103,
            'name': 'GoldCard',
            'category': 'Kreditkarte',
            'risk_class': 'niedrig',
            'price': 150.00,
            'short_desc': 'Premium Kreditkarte mit Versicherungsschutz und weltweiten Vorteilen'
        },
        {
            'prod_id': 104,
            'name': 'LebensSchutz',
            'category': 'Versicherung',
            'risk_class': 'niedrig',
            'price': 280.00,
            'short_desc': 'Umfassende Lebensversicherung mit flexiblen Beitragszahlungen und Gewinnbeteiligung'
        },
        {
            'prod_id': 105,
            'name': 'DepotProfessional',
            'category': 'Depot',
            'risk_class': 'hoch',
            'price': 45.00,
            'short_desc': 'Professionelles Depot für aktive Trader mit erweiterten Analysefunktionen'
        },
        {
            'prod_id': 106,
            'name': 'UnfallSchutz',
            'category': 'Versicherung',
            'risk_class': 'niedrig',
            'price': 120.00,
            'short_desc': 'Umfassender Unfallschutz für Beruf und Freizeit mit schneller Auszahlung'
        }
    ]
    
    return products

def main():
    """Main function to generate and save product data"""
    
    # Generate product data
    products_data = generate_products()
    
    # Create DataFrame
    df = pd.DataFrame(products_data)
    
    # Save to CSV
    df.to_csv('data/products.csv', index=False)
    
    # Success message
    print("products.csv erstellt")
    
    # Optional: Show the created products for verification
    print(f"\n📦 {len(df)} Bankprodukte generiert:")
    for _, product in df.iterrows():
        print(f"   • {product['name']} ({product['category']}) - €{product['price']}/Jahr - {product['risk_class']} Risiko")

if __name__ == "__main__":
    main() 