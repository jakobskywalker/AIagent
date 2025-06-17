#!/usr/bin/env python3
"""
Streamlit App für Bank-Adviser AI
Chat-basierter AI-Assistent für Bankberater
"""

import streamlit as st
import pandas as pd
import numpy as np
from agent_service import recommend, load_data
import warnings
warnings.filterwarnings('ignore')

# Seiten-Konfiguration
st.set_page_config(
    page_title="AI-Agent Demo",
    page_icon="🏦",
    layout="wide"
)

# Cache für Daten
@st.cache_data
def load_all_data():
    """Lade alle CSV-Dateien"""
    customers_df = pd.read_csv('data/customers.csv')
    products_df = pd.read_csv('data/products.csv')
    ownership_df = pd.read_csv('data/ownership.csv')
    sales_df = pd.read_csv('data/sales_history.csv')
    return customers_df, products_df, ownership_df, sales_df

def main():
    # Titel
    st.title("🏦 AI-Agent Demo")
    st.markdown("### Bank-Adviser AI für intelligente Produktempfehlungen")
    
    # Lade Daten
    customers_df, products_df, ownership_df, sales_df = load_all_data()
    
    # Sidebar für Kundenauswahl
    with st.sidebar:
        st.markdown("## 🔧 Einstellungen")
        
        # Kundennummer Eingabe
        st.markdown("### Kundennummer")
        customer_id = st.number_input(
            "Wählen Sie eine Kundennummer:",
            min_value=1,
            max_value=len(customers_df),
            value=1,
            step=1,
            help="Geben Sie eine Kundennummer zwischen 1 und 50 ein"
        )
        
        # Kundeninfo anzeigen
        if customer_id:
            customer = customers_df[customers_df['cust_id'] == customer_id].iloc[0]
            st.markdown("### 👤 Kundeninfo")
            st.markdown(f"**Name:** {customer['first_name']} {customer['last_name']}")
            st.markdown(f"**Ort:** {customer['location']}")
            st.markdown(f"**Alter:** {customer['age_bucket']}")
            st.markdown(f"**Einkommen:** €{customer['revenue']:,.0f}")
            st.markdown(f"**Credit Score:** {customer['credit_score']}")
    
    # Hauptbereich mit Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Empfehlungen", "📊 Snapshot", "❓ Produkt erklären"])
    
    # Tab 1: Empfehlungen
    with tab1:
        st.markdown("## 🎯 Produktempfehlungen")
        st.markdown("Erhalten Sie KI-basierte Produktempfehlungen für den ausgewählten Kunden.")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Anzahl Empfehlungen
            top_k = st.slider("Anzahl Empfehlungen:", 1, 5, 3)
            
            # Button für Empfehlungen
            if st.button("🚀 Empfehlung anfordern", type="primary", use_container_width=True):
                with st.spinner("Generiere Empfehlungen..."):
                    recommendations = recommend(customer_id, top_k=top_k)
                    st.session_state['recommendations'] = recommendations
        
        with col2:
            # Zeige Empfehlungen
            if 'recommendations' in st.session_state and st.session_state['recommendations']:
                st.markdown("### 📋 Empfohlene Produkte")
                
                for i, rec in enumerate(st.session_state['recommendations'], 1):
                    with st.container():
                        col_name, col_score = st.columns([3, 1])
                        
                        with col_name:
                            st.markdown(f"**{i}. {rec['name']}** (Produkt {rec['prod_id']})")
                            st.markdown(f"_{rec['reason']}_")
                        
                        with col_score:
                            # Score als Prozentbalken
                            score_percent = int(rec['score'] * 100)
                            st.metric("Score", f"{score_percent}%")
                            st.progress(rec['score'])
                        
                        st.markdown("---")
                
                # DataFrame-Ansicht
                st.markdown("### 📊 Übersicht als Tabelle")
                df_recommendations = pd.DataFrame(st.session_state['recommendations'])
                df_recommendations['Score (%)'] = (df_recommendations['score'] * 100).round(1)
                df_recommendations = df_recommendations[['name', 'Score (%)', 'reason']]
                df_recommendations.columns = ['Produkt', 'Score (%)', 'Begründung']
                st.dataframe(df_recommendations, use_container_width=True)
            else:
                st.info("👆 Klicken Sie auf 'Empfehlung anfordern' um Produktempfehlungen zu erhalten.")
    
    # Tab 2: Snapshot
    with tab2:
        st.markdown("## 📊 Kunden-Snapshot")
        st.markdown("Übersicht über die aktuellen Produkte und Umsätze des Kunden.")
        
        if st.button("📸 Snapshot anzeigen", type="primary"):
            # Hole Kundenprodukte
            customer_products = ownership_df[ownership_df['cust_id'] == customer_id]
            
            if not customer_products.empty:
                # Merge mit Produktdetails
                customer_products_detailed = customer_products.merge(
                    products_df, 
                    on='prod_id', 
                    how='left'
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📦 Aktuelle Produkte")
                    for _, prod in customer_products_detailed.iterrows():
                        st.markdown(f"• **{prod['name']}** (seit {prod['since_date']})")
                        st.markdown(f"  Kategorie: {prod['category']} | Gebühr: €{prod['price']}/Jahr")
                
                with col2:
                    # Umsatzstatistiken
                    st.markdown("### 💰 Umsatzübersicht")
                    
                    # Hole Umsätze aus sales_history
                    customer_sales = sales_df[sales_df['cust_id'] == customer_id]
                    
                    if not customer_sales.empty:
                        total_revenue = customer_sales['amount'].sum()
                        avg_transaction = customer_sales['amount'].mean()
                        num_transactions = len(customer_sales)
                        
                        st.metric("Gesamtumsatz", f"€{total_revenue:,.2f}")
                        st.metric("Anzahl Transaktionen", num_transactions)
                        st.metric("Ø Transaktionswert", f"€{avg_transaction:,.2f}")
                        
                        # Umsatz pro Produkt
                        st.markdown("### 📈 Umsatz pro Produkt")
                        product_revenue = customer_sales.groupby('prod_id')['amount'].agg(['sum', 'count']).reset_index()
                        product_revenue = product_revenue.merge(products_df[['prod_id', 'name']], on='prod_id')
                        product_revenue.columns = ['Produkt ID', 'Umsatz (€)', 'Anzahl', 'Produktname']
                        
                        for _, row in product_revenue.iterrows():
                            st.markdown(f"**{row['Produktname']}**: €{row['Umsatz (€)']:,.2f} ({row['Anzahl']} Transaktionen)")
                    else:
                        st.info("Keine Transaktionen gefunden.")
                
                # Zusammenfassung
                st.markdown("### 📊 Zusammenfassung")
                total_products = len(customer_products)
                total_annual_fees = customer_products_detailed['price'].sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Anzahl Produkte", total_products)
                with col2:
                    st.metric("Jährliche Gebühren", f"€{total_annual_fees:,.2f}")
                with col3:
                    if not customer_sales.empty:
                        st.metric("Kundenaktivität", f"{num_transactions} Transaktionen")
            else:
                st.warning("Dieser Kunde hat noch keine Produkte.")
    
    # Tab 3: Produkt erklären
    with tab3:
        st.markdown("## ❓ Produkt erklären")
        st.markdown("Erhalten Sie detaillierte Informationen zu unseren Bankprodukten.")
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Produktauswahl
            product_names = products_df['name'].tolist()
            selected_product = st.selectbox(
                "Wählen Sie ein Produkt:",
                options=product_names,
                help="Wählen Sie ein Produkt aus der Liste"
            )
            
            # Oder Produkt-ID eingeben
            st.markdown("**Oder geben Sie eine Produkt-ID ein:**")
            product_id_input = st.text_input(
                "Produkt-ID (z.B. 101):",
                placeholder="101-106"
            )
            
            # Button zum Erklären
            if st.button("📖 Produkt erklären", type="primary", use_container_width=True):
                # Finde Produkt
                if product_id_input:
                    try:
                        prod_id = int(product_id_input)
                        product = products_df[products_df['prod_id'] == prod_id]
                    except:
                        product = pd.DataFrame()
                else:
                    product = products_df[products_df['name'] == selected_product]
                
                if not product.empty:
                    st.session_state['explained_product'] = product.iloc[0].to_dict()
                else:
                    st.error("Produkt nicht gefunden.")
        
        with col2:
            # Zeige Produkterklärung
            if 'explained_product' in st.session_state:
                prod = st.session_state['explained_product']
                
                st.markdown(f"### 📦 {prod['name']}")
                st.markdown(f"**Produkt-ID:** {prod['prod_id']}")
                
                # Produktdetails in schöner Box
                with st.container():
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**Kategorie:** {prod['category']}")
                        st.markdown(f"**Risikoklasse:** {prod['risk_class']}")
                    with col_b:
                        st.markdown(f"**Jahresgebühr:** €{prod['price']}")
                        
                        # Risiko-Indikator
                        risk_colors = {'niedrig': '🟢', 'mittel': '🟡', 'hoch': '🔴'}
                        risk_icon = risk_colors.get(prod['risk_class'], '⚪')
                        st.markdown(f"**Risiko:** {risk_icon} {prod['risk_class']}")
                
                # Kurzbeschreibung
                st.markdown("### 📝 Beschreibung")
                st.info(prod['short_desc'])
                
                # Zusätzliche Infos basierend auf Kategorie
                st.markdown("### 💡 Weitere Informationen")
                
                category_info = {
                    'Giro': "Girokonto für den täglichen Zahlungsverkehr mit verschiedenen Zusatzleistungen.",
                    'Depot': "Wertpapierdepot für Aktien, Fonds und andere Anlageprodukte.",
                    'Kreditkarte': "Kreditkarte für weltweite Zahlungen mit zusätzlichen Services.",
                    'Versicherung': "Absicherung gegen verschiedene Lebensrisiken."
                }
                
                st.markdown(category_info.get(prod['category'], "Bankprodukt mit speziellen Features."))
            else:
                st.info("👆 Wählen Sie ein Produkt und klicken Sie auf 'Produkt erklären'.")
    
    # Footer
    st.markdown("---")
    st.markdown("🏦 **Bank-Adviser AI** | Powered by Machine Learning & AI | Alle Daten bleiben lokal")

if __name__ == "__main__":
    main() 