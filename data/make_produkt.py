#!/usr/bin/env python3
"""
Script to generate bank product data.
Creates 10 bank products including real estate financing products and saves them as products.csv.
"""

import pandas as pd

def generate_products():
    """Generate 10 bank product data entries including real estate financing"""
    
    products = [
        {
            'prod_id': 101,
            'name': 'GiroPlus',
            'category': 'Giro',
            'risk_class': 'niedrig',
            'price': 89.90,
            'interest_rate': None,  # Keine Zinsen bei Girokonten
            'short_desc': 'Premium Girokonto mit kostenlosen Überweisungen und Kreditkarte inklusive'
        },
        {
            'prod_id': 102,
            'name': 'DepotBasic',
            'category': 'Depot',
            'risk_class': 'mittel',
            'price': 12.50,
            'interest_rate': None,  # Keine Zinsen bei Depots
            'short_desc': 'Günstige Wertpapierdepot für Einsteiger mit reduzierten Ordergebühren'
        },
        {
            'prod_id': 103,
            'name': 'GoldCard',
            'category': 'Kreditkarte',
            'risk_class': 'niedrig',
            'price': 150.00,
            'interest_rate': None,  # Keine Zinsen bei Kreditkarten
            'short_desc': 'Premium Kreditkarte mit Versicherungsschutz und weltweiten Vorteilen'
        },
        {
            'prod_id': 104,
            'name': 'LebensSchutz',
            'category': 'Versicherung',
            'risk_class': 'niedrig',
            'price': 280.00,
            'interest_rate': None,  # Keine Zinsen bei Versicherungen
            'short_desc': 'Umfassende Lebensversicherung mit flexiblen Beitragszahlungen und Gewinnbeteiligung'
        },
        {
            'prod_id': 105,
            'name': 'DepotProfessional',
            'category': 'Depot',
            'risk_class': 'hoch',
            'price': 45.00,
            'interest_rate': None,  # Keine Zinsen bei Depots
            'short_desc': 'Professionelles Depot für aktive Trader mit erweiterten Analysefunktionen'
        },
        {
            'prod_id': 106,
            'name': 'UnfallSchutz',
            'category': 'Versicherung',
            'risk_class': 'niedrig',
            'price': 120.00,
            'interest_rate': None,  # Keine Zinsen bei Versicherungen
            'short_desc': 'Umfassender Unfallschutz für Beruf und Freizeit mit schneller Auszahlung'
        },
        # Neue Immobilienfinanzierungsprodukte
        {
            'prod_id': 107,
            'name': 'BauFinanz Standard',
            'category': 'Immobilienkredit',
            'risk_class': 'mittel',
            'price': 350.00,  # Jährliche Kontoführungsgebühr
            'interest_rate': 3.85,  # Basis-Zinssatz (individuell angepasst in Kunde_Produkt)
            'short_desc': 'Klassische Baufinanzierung mit flexiblen Laufzeiten von 10-30 Jahren und Sondertilgungsoption (Zinssatz ab 3,85% p.a.)'
        },
        {
            'prod_id': 108,
            'name': 'WohnTraum Flex',
            'category': 'Immobilienkredit',
            'risk_class': 'mittel',
            'price': 450.00,  # Jährliche Kontoführungsgebühr
            'interest_rate': 4.25,  # Basis-Zinssatz (individuell angepasst in Kunde_Produkt)
            'short_desc': 'Flexible Immobilienfinanzierung mit variablem Zinssatz und kostenloser Ratenpausenoption (Zinssatz ab 4,25% p.a.)'
        },
        {
            'prod_id': 109,
            'name': 'ImmoInvest Pro',
            'category': 'Immobilienkredit',
            'risk_class': 'hoch',
            'price': 550.00,  # Jährliche Kontoführungsgebühr
            'interest_rate': 4.95,  # Basis-Zinssatz (individuell angepasst in Kunde_Produkt)
            'short_desc': 'Finanzierung für Kapitalanleger mit bis zu 100% Beleihung und steuerlichen Vorteilen (Zinssatz ab 4,95% p.a.)'
        },
        {
            'prod_id': 110,
            'name': 'ErstHeim Bonus',
            'category': 'Immobilienkredit',
            'risk_class': 'niedrig',
            'price': 250.00,  # Reduzierte Gebühr für Erstkäufer
            'interest_rate': 3.45,  # Basis-Zinssatz (individuell angepasst in Kunde_Produkt)
            'short_desc': 'Spezialfinanzierung für Erstkäufer mit staatlicher Förderung und reduzierten Zinsen (Zinssatz ab 3,45% p.a.)'
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
    df.to_csv('products.csv', index=False)
    
    # Success message
    print("products.csv erstellt")
    
    # Optional: Show the created products for verification
    print(f"\n📦 {len(df)} Bankprodukte generiert:")
    for _, product in df.iterrows():
        print(f"   • {product['name']} ({product['category']}) - €{product['price']}/Jahr - {product['risk_class']} Risiko")

if __name__ == "__main__":
    main() 