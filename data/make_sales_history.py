#!/usr/bin/env python3
"""
Script to generate sales history data.
Loads ownership.csv and simulates 0-4 transactions for each customer-product pair.
"""

import pandas as pd
import random
import uuid
from datetime import datetime, timedelta

def generate_random_date_between(start_date_str, end_date_str):
    """Generate a random date between start_date and end_date"""
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    # Calculate the difference in days
    time_between = end_date - start_date
    days_between = time_between.days
    
    # Generate random number of days
    random_days = random.randint(0, max(0, days_between))
    random_date = start_date + timedelta(days=random_days)
    
    return random_date.strftime('%Y-%m-%d')

def generate_sales_history():
    """Generate sales history for all customer-product relationships"""
    
    # Load ownership data
    ownership_df = pd.read_csv('data/ownership.csv')
    
    sales_records = []
    today = datetime.now().strftime('%Y-%m-%d')
    
    # For each customer-product relationship
    for _, ownership in ownership_df.iterrows():
        cust_id = ownership['cust_id']
        prod_id = ownership['prod_id']
        since_date = ownership['since_date']
        
        # Generate 0-4 transactions for this customer-product pair
        num_transactions = random.randint(0, 4)
        
        for _ in range(num_transactions):
            # Generate transaction details
            sale_record = {
                'sale_id': str(uuid.uuid4()),
                'cust_id': cust_id,
                'prod_id': prod_id,
                'amount': round(random.uniform(100, 10000), 2),
                'sale_date': generate_random_date_between(since_date, today)
            }
            sales_records.append(sale_record)
    
    return sales_records

def main():
    """Main function to generate and save sales history data"""
    
    # Generate sales history data
    sales_data = generate_sales_history()
    
    # Create DataFrame
    df = pd.DataFrame(sales_data)
    
    # Sort by customer ID, product ID, and sale date for better readability
    if not df.empty:
        df = df.sort_values(['cust_id', 'prod_id', 'sale_date']).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv('data/sales_history.csv', index=False)
    
    # Success message
    print(f'sales_history.csv erstellt, {len(df)} Zeilen')
    
    # Optional: Show some statistics
    if not df.empty:
        print(f"\n📊 Verkaufshistorie Übersicht:")
        print(f"   • Gesamtumsatz: €{df['amount'].sum():,.2f}")
        print(f"   • Durchschnittlicher Transaktionswert: €{df['amount'].mean():,.2f}")
        print(f"   • Transaktionen pro Kunde-Produkt-Paar: {len(df) / len(pd.read_csv('data/ownership.csv')):.1f}")
        print(f"   • Anzahl Kunden mit Transaktionen: {df['cust_id'].nunique()}")
        print(f"   • Anzahl Produkte mit Transaktionen: {df['prod_id'].nunique()}")
        
        print(f"\n💰 Umsatz nach Produkten:")
        product_sales = df.groupby('prod_id')['amount'].agg(['count', 'sum']).reset_index()
        product_sales.columns = ['prod_id', 'anzahl_transaktionen', 'gesamtumsatz']
        for _, row in product_sales.iterrows():
            print(f"   • Produkt {row['prod_id']}: {row['anzahl_transaktionen']} Transaktionen, €{row['gesamtumsatz']:,.2f}")
    else:
        print("   • Keine Transaktionen generiert")

if __name__ == "__main__":
    main() 