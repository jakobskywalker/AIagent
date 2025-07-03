#!/usr/bin/env python3
"""
Streamlit App für Bank-Adviser AI
Chat-basierter AI-Assistent für Bankberater
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from agent_service import recommend, load_data, top_potential_analysis
from agent_llm import chat_llm
import chatlog_service
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# OpenAI API Key aus Umgebungsvariable oder Session State
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = os.getenv('OPENAI_API_KEY', '')

# Seiten-Konfiguration
st.set_page_config(
    page_title="AI-Agent Demo",
    page_icon="🏦",
    layout="wide"
)

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize current customer ID
if 'current_customer_id' not in st.session_state:
    st.session_state.current_customer_id = None

# Initialize auto-save preference
if 'auto_save_chats' not in st.session_state:
    st.session_state.auto_save_chats = True

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
        
        # Auto-Save Option
        st.markdown("### 💾 Chat-Einstellungen")
        st.session_state.auto_save_chats = st.checkbox(
            "Chats automatisch speichern",
            value=st.session_state.auto_save_chats,
            help="Speichert Chats automatisch, wenn Sie den Kunden wechseln"
        )
        
        # Wenn Kunde gewechselt wurde und Auto-Save aktiv ist
        if st.session_state.current_customer_id and st.session_state.current_customer_id != customer_id:
            if st.session_state.auto_save_chats and st.session_state.chat_history:
                # Speichere den Chat des vorherigen Kunden
                chatlog_service.save_chatlog(
                    customer_id=st.session_state.current_customer_id,
                    chat_history=st.session_state.chat_history,
                    metadata={
                        'saved_at': datetime.now().isoformat(),
                        'api_mode': 'OpenAI' if st.session_state.openai_api_key else 'Mock',
                        'auto_saved': True
                    }
                )
                st.session_state.chat_history = []
        
        # Aktualisiere aktuelle Kunden-ID
        st.session_state.current_customer_id = customer_id
        
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎯 Empfehlungen", "📊 Snapshot", "❓ Produkt erklären", "💬 Chat mit AI", "📜 Chat-Historie", "💎 Top Potential"])
    
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

                            # XAI Expander
                            if rec.get('contributions'):
                                with st.expander("🧠 XAI Details", expanded=False):
                                    top_contribs = rec['contributions'][:5]
                                    df_contrib = pd.DataFrame(top_contribs)
                                    df_contrib_display = df_contrib[['feature', 'contribution']].copy()
                                    df_contrib_display['Contribution'] = df_contrib_display['contribution'].round(3)
                                    df_contrib_display = df_contrib_display[['feature', 'Contribution']]
                                    st.bar_chart(df_contrib_display.set_index('feature'))
                                    st.caption("Positiv = erhöht Abschlusswahrscheinlichkeit, negativ = verringert")
                        
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
    
    # Tab 4: Chat mit AI
    with tab4:
        st.markdown("## 💬 Chat mit Bank-Berater AI")
        st.markdown("Stellen Sie Fragen zu Bankprodukten, Empfehlungen und dem Kundenprofil.")
        st.markdown("Der AI-Agent kann automatisch Tools nutzen: Empfehlungen, Snapshots, Produkterklärungen.")
        
        # API Key Settings in Sidebar
        with st.sidebar:
            st.markdown("### 🔑 API Einstellungen")
            api_key_input = st.text_input(
                "OpenAI API Key:",
                value=st.session_state.openai_api_key,
                type="password",
                help="Geben Sie Ihren OpenAI API Key ein für erweiterte Chat-Funktionen. Lassen Sie das Feld leer für Mock-Modus."
            )
            if api_key_input != st.session_state.openai_api_key:
                st.session_state.openai_api_key = api_key_input
                if api_key_input:
                    st.success("API Key aktualisiert!")
                else:
                    st.info("Mock-Modus aktiviert (kein API Key)")
            
            # Status Anzeige
            if st.session_state.openai_api_key:
                st.markdown("🟢 **Status:** OpenAI API")
            else:
                st.markdown("🔵 **Status:** Mock-Modus")
            
            # Zeige Kunden mit Chatlogs
            st.markdown("### 📚 Kunden mit Chatlogs")
            customers_with_logs = chatlog_service.get_all_customers_with_chatlogs()
            if customers_with_logs:
                st.info(f"{len(customers_with_logs)} Kunden haben gespeicherte Chats")
                st.caption("Kunden-IDs: " + ", ".join(map(str, customers_with_logs[:10])) + ("..." if len(customers_with_logs) > 10 else ""))
        
        # Chat Interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    with st.chat_message('user'):
                        st.write(message['content'])
                else:
                    with st.chat_message('assistant', avatar='🏦'):
                        st.write(message['content'])
        
        # Chat input mit agent_llm
        user_msg = st.chat_input("Frag den Agenten ...")
        
        if user_msg:
            # Add user message to history
            st.session_state.chat_history.append({'role': 'user', 'content': user_msg})
            
            # Get AI response using agent_llm
            with st.spinner("AI denkt nach..."):
                answer = chat_llm(
                    user_msg=user_msg,
                    api_key=st.session_state.openai_api_key,
                    customer_id=customer_id,
                    chat_history=st.session_state.chat_history
                )
            
            # Add AI response to history
            st.session_state.chat_history.append({'role': 'assistant', 'content': answer})
            
            # Show response with chat_message
            with st.chat_message("assistant", avatar='🏦'):
                st.write(answer)
            
            # Rerun to update chat display
            st.rerun()
        
        # Clear chat button and save button
        col1, col2, col3 = st.columns([4, 1, 1])
        with col2:
            if st.button("💾 Speichern") and st.session_state.chat_history:
                # Speichere aktuellen Chat
                chatlog_service.save_chatlog(
                    customer_id=customer_id,
                    chat_history=st.session_state.chat_history,
                    metadata={
                        'saved_at': datetime.now().isoformat(),
                        'api_mode': 'OpenAI' if st.session_state.openai_api_key else 'Mock'
                    }
                )
                st.success("Chat gespeichert!")
                st.session_state.chat_history = []
                st.rerun()
        
        with col3:
            if st.button("🗑️ Chat löschen"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Example prompts
        st.markdown("### 💡 Beispiel-Fragen:")
        example_prompts = [
            f"Welche Produkte schlagen wir Kunde {customer_id} vor?",
            f"Zeig mir bitte den Snapshot von Kunde {customer_id}.",
            "Was kostet DepotPlus?",
            "Liste alle verfügbaren Produkte auf.",
            f"Warum sollte Kunde {customer_id} ein Depot eröffnen?"
        ]
        
        cols = st.columns(2)
        for i, prompt in enumerate(example_prompts):
            with cols[i % 2]:
                if st.button(prompt, key=f"example_{i}"):
                    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
                    with st.spinner("AI denkt nach..."):
                        answer = chat_llm(
                            user_msg=prompt,
                            api_key=st.session_state.openai_api_key,
                            customer_id=customer_id,
                            chat_history=st.session_state.chat_history
                        )
                    st.session_state.chat_history.append({'role': 'assistant', 'content': answer})
                    st.rerun()
    
    # Tab 5: Chat-Historie
    with tab5:
        st.markdown("## 📜 Chat-Historie")
        st.markdown(f"Gespeicherte Chat-Verläufe für {customer['first_name']} {customer['last_name']} (Kunde {customer_id})")
        
        # Lade alle Chatlogs für den Kunden
        chatlogs = chatlog_service.load_all_chatlogs(customer_id)
        
        if chatlogs:
            # Statistiken anzeigen
            stats = chatlog_service.get_chatlog_statistics(customer_id)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Gespeicherte Chats", stats['total_sessions'])
            with col2:
                st.metric("Gesamte Nachrichten", stats['total_messages'])
            with col3:
                st.metric("Ø Nachrichten pro Chat", f"{stats['avg_messages_per_session']:.1f}")
            
            # Suchfunktion
            st.markdown("### 🔍 Chatlogs durchsuchen")
            search_term = st.text_input("Suchbegriff eingeben:", placeholder="z.B. DepotBasic, Empfehlung, Kosten...")
            
            # Filter Chatlogs basierend auf Suche
            if search_term:
                filtered_logs = chatlog_service.search_chatlogs(customer_id, search_term)
                st.info(f"Gefunden in {len(filtered_logs)} von {len(chatlogs)} Chats")
            else:
                filtered_logs = chatlogs
            
            # Zeige Chatlogs
            st.markdown("### 💬 Gespeicherte Chats")
            
            # Sortiere nach Datum (neueste zuerst)
            filtered_logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            for log in filtered_logs:
                with st.expander(f"Chat vom {log['timestamp'][:19].replace('T', ' ')} - {log['message_count']} Nachrichten"):
                    # Zeige Metadaten
                    metadata_cols = st.columns(3)
                    with metadata_cols[0]:
                        st.caption(f"📅 Session: {log.get('session_id', 'Unbekannt')}")
                    with metadata_cols[1]:
                        st.caption(f"🤖 Modus: {log.get('metadata', {}).get('api_mode', 'Unbekannt')}")
                    with metadata_cols[2]:
                        auto_saved = log.get('metadata', {}).get('auto_saved', False)
                        st.caption(f"💾 {'Auto-gespeichert' if auto_saved else 'Manuell gespeichert'}")
                    
                    # Zeige Nachrichten
                    for msg in log['messages']:
                        if msg['role'] == 'user':
                            st.markdown(f"**👤 Kunde:** {msg['content']}")
                        else:
                            st.markdown(f"**🏦 AI-Agent:** {msg['content']}")
                    
                    # Aktionen für diesen Chatlog
                    col_a, col_b = st.columns([5, 1])
                    with col_b:
                        if st.button("🗑️", key=f"del_{log['session_id']}", help="Diesen Chat löschen"):
                            if chatlog_service.delete_chatlog(customer_id, log['session_id']):
                                st.success("Chat gelöscht!")
                                st.rerun()
            
            # Export-Funktionen
            st.markdown("### 📤 Export-Optionen")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Als CSV exportieren"):
                    export_path = f"data/chatlogs/export_customer_{customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    chatlog_service.export_chatlogs_to_csv(customer_id, export_path)
                    st.success(f"Exportiert nach: {export_path}")
            
            with col2:
                if st.button("🗑️ Alle Chats löschen", type="secondary"):
                    if chatlog_service.delete_all_chatlogs(customer_id):
                        st.success("Alle Chats gelöscht!")
                        st.rerun()
        
        else:
            st.info("📭 Noch keine gespeicherten Chats für diesen Kunden vorhanden.")
            st.markdown("💡 **Tipp:** Führen Sie einen Chat im 'Chat mit AI' Tab und klicken Sie auf 'Speichern' um den Verlauf zu sichern.")
    
    # Tab 6: Top Potential Analyse
    with tab6:
        st.markdown("## 💎 Top Potential Analyse")
        st.markdown("Ermittelt den Kunden mit dem höchsten erwarteten Gewinn basierend auf Produktempfehlungen für alle Kunden.")

        if st.button("🔎 Analyse starten", type="primary", use_container_width=True):
            # Fortschrittsanzeige / Animation
            progress_bar = st.progress(0)
            status_placeholder = st.empty()

            for pct in range(0, 101, 10):
                progress_bar.progress(pct)
                status_placeholder.text(f"Analysiere Daten... {pct}%")
                time.sleep(0.05)

            with st.spinner("Berechne Top-Potential-Analyse..."):
                results = top_potential_analysis(top_k_per_customer=3)

            if results:
                best = results[0]
                best_customer = customers_df[customers_df['cust_id'] == best['cust_id']].iloc[0]

                st.success(f"💎 Top-Kunde: {best_customer['first_name']} {best_customer['last_name']} (ID {best['cust_id']})")
                st.metric("Erwarteter Gewinn", f"€{best['expected_profit']:.2f}")

                # Details der Empfehlungen
                st.markdown("### 📦 Empfohlene Produkte & erwarteter Gewinn")
                df_recs = pd.DataFrame(best['recommendations'])
                if not df_recs.empty:
                    df_recs['Score (%)'] = (df_recs['score'] * 100).round(1)
                    df_recs['Gewinn (€)'] = df_recs['expected_profit'].round(2)
                    df_recs = df_recs[['name', 'Score (%)', 'price', 'Gewinn (€)']]
                    df_recs.columns = ['Produkt', 'Score (%)', 'Preis (€)', 'Gewinn (€)']
                    st.dataframe(df_recs, use_container_width=True)
                else:
                    st.info("Keine Empfehlungen für diesen Kunden verfügbar.")
            else:
                st.warning("Keine Daten für die Analyse gefunden.")

    # --- Footer ---
    st.markdown("---")
    st.markdown("🏦 **Bank-Adviser AI** | Powered by Machine Learning & AI | Alle Daten bleiben lokal")

if __name__ == "__main__":
    main() 