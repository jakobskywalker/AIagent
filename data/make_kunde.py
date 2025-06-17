#!/usr/bin/env python3
"""
Script to generate synthetic customer data using faker library.
Creates 50 customers and saves them as customers.csv.
"""

import pandas as pd
from faker import Faker
import random
from datetime import datetime, date

def calculate_age_bucket(birth_date):
    """Calculate age bucket from birth date"""
    today = date.today()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    
    if 18 <= age <= 24:
        return "18-24"
    elif 25 <= age <= 39:
        return "25-39"
    elif 40 <= age <= 59:
        return "40-59"
    else:
        return "60+"

def generate_customers(num_customers=50):
    """Generate synthetic customer data"""
    
    # Initialize faker with German locale for realistic names and cities
    fake = Faker('de_DE')
    
    customers = []
    
    for i in range(1, num_customers + 1):
        # Generate birth date for realistic age distribution
        birth_date = fake.date_of_birth(minimum_age=18, maximum_age=80)
        
        customer = {
            'cust_id': i,
            'first_name': fake.first_name(),
            'last_name': fake.last_name(),
            'location': fake.city(),
            'revenue': random.randint(20000, 250000),
            'credit_score': random.randint(300, 850),
            'age_bucket': calculate_age_bucket(birth_date)
        }
        
        customers.append(customer)
    
    return customers

def main():
    """Main function to generate and save customer data"""
    
    print("🏦 Generiere synthetische Kundendaten...")
    
    # Generate customer data
    customers_data = generate_customers(50)
    
    # Create DataFrame
    df = pd.DataFrame(customers_data)
    
    # Save to CSV
    df.to_csv('data/customers.csv', index=False)
    
    # Display summary statistics
    print(f"\n✅ Erfolgreich {len(df)} Kunden generiert und in 'data/customers.csv' gespeichert!")
    print(f"\n📊 Zusammenfassung:")
    print(f"   • Anzahl Kunden: {len(df)}")
    print(f"   • Durchschnittliches Einkommen: €{df['revenue'].mean():,.0f}")
    print(f"   • Durchschnittlicher Credit Score: {df['credit_score'].mean():.0f}")
    print(f"\n🎯 Altersverteilung:")
    age_distribution = df['age_bucket'].value_counts().sort_index()
    for age_bucket, count in age_distribution.items():
        print(f"   • {age_bucket}: {count} Kunden ({count/len(df)*100:.1f}%)")
    
    print(f"\n🏙️ Top 5 Städte:")
    city_distribution = df['location'].value_counts().head(5)
    for city, count in city_distribution.items():
        print(f"   • {city}: {count} Kunden")
    
    print(f"\n💰 Einkommensverteilung:")
    print(f"   • Min: €{df['revenue'].min():,}")
    print(f"   • Max: €{df['revenue'].max():,}")
    print(f"   • Median: €{df['revenue'].median():,}")
    
    print(f"\n📈 Credit Score Verteilung:")
    print(f"   • Min: {df['credit_score'].min()}")
    print(f"   • Max: {df['credit_score'].max()}")
    print(f"   • Median: {df['credit_score'].median()}")
    
    # Show first few rows
    print(f"\n📋 Erste 5 Kunden:")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    main() 