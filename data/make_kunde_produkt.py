#!/usr/bin/env python3
"""
Script to generate customer-product ownership relationships.
Loads customers.csv and products.csv, assigns 1-3 random products to each customer.
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

def generate_ownership():
    """Generate customer-product ownership relationships"""
    
    # Load existing data
    customers_df = pd.read_csv('data/customers.csv')
    products_df = pd.read_csv('data/products.csv')
    
    ownership_records = []
    
    # Get all available product IDs
    available_products = products_df['prod_id'].tolist()
    
    # For each customer, assign 1-3 random products
    for _, customer in customers_df.iterrows():
        customer_id = customer['cust_id']
        
        # Randomly choose how many products this customer will have (1-3)
        num_products = random.randint(1, 3)
        
        # Randomly select products without duplicates
        selected_products = random.sample(available_products, num_products)
        
        # Create ownership records for this customer
        for product_id in selected_products:
            ownership_record = {
                'cust_id': customer_id,
                'prod_id': product_id,
                'since_date': generate_random_date()
            }
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
    df.to_csv('data/ownership.csv', index=False)
    
    # Success message with exact format requested
    print(f'ownership.csv erstellt, {len(df)} Zeilen')
    
    # Optional: Show some statistics
    print(f"\n📊 Übersicht:")
    print(f"   • Durchschnittliche Produkte pro Kunde: {len(df) / df['cust_id'].nunique():.1f}")
    print(f"   • Produktverteilung:")
    product_counts = df['prod_id'].value_counts().sort_index()
    for prod_id, count in product_counts.items():
        print(f"     - Produkt {prod_id}: {count} Kunden")

if __name__ == "__main__":
    main() 