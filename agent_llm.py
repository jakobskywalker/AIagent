#!/usr/bin/env python3
"""
Agent LLM mit OpenAI Function-Calling für Bank-Adviser AI
Nutzt LangChain für Tool-Integration mit recommend, snapshot, explain
"""

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_core.messages import SystemMessage
import pandas as pd
from typing import Dict, List, Optional
from agent_service import recommend, load_data
import json
import re

# Lade Daten einmalig
_customers_df = None
_products_df = None
_ownership_df = None
_sales_df = None

def get_data():
    """Lade alle benötigten DataFrames"""
    global _customers_df, _products_df, _ownership_df, _sales_df
    
    if _customers_df is None:
        _customers_df = pd.read_csv('data/customers.csv')
        _products_df = pd.read_csv('data/products.csv')
        _ownership_df = pd.read_csv('data/ownership.csv')
        _sales_df = pd.read_csv('data/sales_history.csv')
    
    return _customers_df, _products_df, _ownership_df, _sales_df

@tool
def recommend_products(customer_id: int, top_k: int = 3) -> str:
    """
    Empfiehlt die Top-k Produkte für einen Kunden basierend auf ML-Modell.
    
    Args:
        customer_id: Die ID des Kunden (1-50)
        top_k: Anzahl der Empfehlungen (default: 3)
    
    Returns:
        String mit formatierten Produktempfehlungen
    """
    try:
        recommendations = recommend(customer_id, top_k=top_k)
        
        if not recommendations:
            return f"Keine Empfehlungen für Kunde {customer_id} gefunden."
        
        result = f"🎯 Top {top_k} Produktempfehlungen für Kunde {customer_id}:\n\n"
        for i, rec in enumerate(recommendations, 1):
            result += f"{i}. **{rec['name']}** (Produkt {rec['prod_id']})\n"
            result += f"   Score: {rec['score']:.2%}\n"
            result += f"   Begründung: {rec['reason']}\n\n"
        
        return result
    except Exception as e:
        return f"Fehler bei Empfehlungen: {str(e)}"

@tool
def get_customer_snapshot(customer_id: int) -> str:
    """
    Zeigt eine Übersicht über Kundenprofile, Produkte und Umsätze.
    
    Args:
        customer_id: Die ID des Kunden (1-50)
    
    Returns:
        String mit formatiertem Kunden-Snapshot
    """
    try:
        customers_df, products_df, ownership_df, sales_df = get_data()
        
        # Hole Kundendaten
        customer = customers_df[customers_df['cust_id'] == customer_id]
        if customer.empty:
            return f"Kunde {customer_id} nicht gefunden."
        
        customer = customer.iloc[0]
        
        # Kundeninfo
        result = f"📊 **Kunden-Snapshot für {customer['first_name']} {customer['last_name']}**\n\n"
        result += f"📍 **Stammdaten:**\n"
        result += f"- Ort: {customer['location']}\n"
        result += f"- Alter: {customer['age_bucket']}\n"
        result += f"- Einkommen: €{customer['revenue']:,.0f}\n"
        result += f"- Credit Score: {customer['credit_score']}\n\n"
        
        # Produkte
        customer_products = ownership_df[ownership_df['cust_id'] == customer_id]
        if not customer_products.empty:
            customer_products_detailed = customer_products.merge(products_df, on='prod_id', how='left')
            
            result += f"📦 **Aktuelle Produkte ({len(customer_products)} Stück):**\n"
            total_fees = 0
            for _, prod in customer_products_detailed.iterrows():
                result += f"- {prod['name']} (seit {prod['since_date']})\n"
                result += f"  Kategorie: {prod['category']} | Gebühr: €{prod['price']}/Jahr\n"
                total_fees += prod['price']
            
            result += f"\n💰 **Jährliche Gesamtgebühren:** €{total_fees:,.2f}\n"
        else:
            result += "📦 **Keine Produkte vorhanden**\n"
        
        # Umsätze
        customer_sales = sales_df[sales_df['cust_id'] == customer_id]
        if not customer_sales.empty:
            total_revenue = customer_sales['amount'].sum()
            num_transactions = len(customer_sales)
            avg_transaction = customer_sales['amount'].mean()
            
            result += f"\n💳 **Transaktionsübersicht:**\n"
            result += f"- Gesamtumsatz: €{total_revenue:,.2f}\n"
            result += f"- Anzahl Transaktionen: {num_transactions}\n"
            result += f"- Ø Transaktionswert: €{avg_transaction:,.2f}\n"
        
        return result
        
    except Exception as e:
        return f"Fehler beim Snapshot: {str(e)}"

@tool
def explain_product(product_identifier: str) -> str:
    """
    Erklärt ein Bankprodukt im Detail. Kann Produkt-ID oder Produktname sein.
    
    Args:
        product_identifier: Produkt-ID (z.B. "102") oder Produktname (z.B. "DepotBasic")
    
    Returns:
        String mit detaillierter Produkterklärung
    """
    try:
        _, products_df, _, _ = get_data()
        
        # Versuche zuerst als ID
        try:
            prod_id = int(product_identifier)
            product = products_df[products_df['prod_id'] == prod_id]
        except:
            # Sonst als Name
            product = products_df[products_df['name'].str.lower() == product_identifier.lower()]
        
        if product.empty:
            return f"Produkt '{product_identifier}' nicht gefunden. Verfügbare Produkte: " + \
                   ", ".join(products_df['name'].tolist())
        
        prod = product.iloc[0]
        
        result = f"📦 **{prod['name']}** (Produkt {prod['prod_id']})\n\n"
        result += f"📋 **Details:**\n"
        result += f"- Kategorie: {prod['category']}\n"
        result += f"- Risikoklasse: {prod['risk_class']}\n"
        result += f"- Jahresgebühr: €{prod['price']}\n\n"
        result += f"📝 **Beschreibung:**\n{prod['short_desc']}\n\n"
        
        # Zusätzliche Infos je nach Kategorie
        category_info = {
            'Giro': "💳 Girokonto für den täglichen Zahlungsverkehr mit verschiedenen Zusatzleistungen.",
            'Depot': "📈 Wertpapierdepot für Aktien, Fonds und andere Anlageprodukte.",
            'Kreditkarte': "💳 Kreditkarte für weltweite Zahlungen mit zusätzlichen Services.",
            'Versicherung': "🛡️ Absicherung gegen verschiedene Lebensrisiken."
        }
        
        if prod['category'] in category_info:
            result += f"💡 **Kategorie-Info:**\n{category_info[prod['category']]}"
        
        return result
        
    except Exception as e:
        return f"Fehler bei Produkterklärung: {str(e)}"

@tool
def list_all_products() -> str:
    """
    Listet alle verfügbaren Bankprodukte auf.
    
    Returns:
        String mit Übersicht aller Produkte
    """
    try:
        _, products_df, _, _ = get_data()
        
        result = "📋 **Verfügbare Bankprodukte:**\n\n"
        
        for _, prod in products_df.iterrows():
            result += f"• **{prod['name']}** (ID: {prod['prod_id']})\n"
            result += f"  {prod['category']} | €{prod['price']}/Jahr | Risiko: {prod['risk_class']}\n"
            result += f"  {prod['short_desc']}\n\n"
        
        return result
        
    except Exception as e:
        return f"Fehler beim Auflisten der Produkte: {str(e)}"

@tool
def top_potential_customer(top_k: int = 3) -> str:
    """
    Liefert den Kunden mit dem höchsten erwarteten Gewinn basierend auf Produktempfehlungen.

    Args:
        top_k: Anzahl der Empfehlungen pro Kunde, die in die Berechnung einfließen.

    Returns:
        Formatierten Text mit Top-Potential-Kunde, erwarteten Gewinn und empfohlene Produkte.
    """
    try:
        from agent_service import top_potential_analysis, load_data

        all_results = top_potential_analysis(top_k_per_customer=top_k)
        if not all_results:
            return "Keine Daten verfügbar, um eine Top-Potential-Analyse durchzuführen."

        best = all_results[0]
        cust_id = best['cust_id']
        exp_profit = best['expected_profit']

        # Kundendaten holen für Namen
        customers_df, _, _, _ = get_data()
        customer = customers_df[customers_df['cust_id'] == cust_id]
        if not customer.empty:
            customer_name = f"{customer.iloc[0]['first_name']} {customer.iloc[0]['last_name']}"
        else:
            customer_name = f"Kunde {cust_id}"

        result = f"💎 **Top-Potential-Kunde:** {customer_name} (ID: {cust_id})\n"
        result += f"💶 **Erwarteter Gewinn:** €{exp_profit:,.2f}\n\n"
        result += "📦 **Empfohlene Produkte:**\n"
        for rec in best['recommendations']:
            result += f"- {rec['name']} (ID {rec['prod_id']}) – Score {rec['score']:.2%} – Erwarteter Gewinn €{rec['expected_profit']:.2f}\n"
        return result
    except Exception as e:
        return f"Fehler bei Top-Potential-Analyse: {str(e)}"

def mock_chat_response(user_msg: str, customer_id: int = None) -> str:
    """
    Mock-Antworten ohne OpenAI API - parsed die Nachricht und ruft die passende Funktion auf
    """
    msg_lower = user_msg.lower()
    
    # Extrahiere Zahlen aus der Nachricht für Kunden-IDs
    numbers = re.findall(r'\d+', user_msg)
    extracted_customer_id = int(numbers[0]) if numbers else customer_id
    
    # Empfehlungen
    if any(word in msg_lower for word in ['empfehl', 'vorschlag', 'schlagen', 'recommend']):
        if extracted_customer_id:
            return recommend_products.invoke({"customer_id": extracted_customer_id, "top_k": 3})
        else:
            return "Bitte geben Sie eine Kunden-ID an (z.B. 'Welche Produkte empfehlen Sie Kunde 17?')"
    
    # Snapshot
    elif any(word in msg_lower for word in ['snapshot', 'übersicht', 'profil', 'zeig mir']):
        if extracted_customer_id:
            return get_customer_snapshot.invoke({"customer_id": extracted_customer_id})
        else:
            return "Bitte geben Sie eine Kunden-ID an (z.B. 'Zeigen Sie mir den Snapshot von Kunde 23')"
    
    # Produkt erklären
    elif any(word in msg_lower for word in ['erkläre', 'was ist', 'was kostet', 'preis', 'kosten']):
        # Suche nach Produktnamen
        products = ['giroplus', 'depotbasic', 'goldcard', 'lebensschutz', 'depotprofessional', 'unfallschutz', 'depotplus']
        for product in products:
            if product in msg_lower:
                return explain_product.invoke({"product_identifier": product})
        
        # Suche nach Produkt-IDs
        if numbers:
            for num in numbers:
                if 101 <= int(num) <= 106:
                    return explain_product.invoke({"product_identifier": num})
        
        return "Bitte nennen Sie ein spezifisches Produkt (z.B. 'Was kostet DepotPlus?')"
    
    # Alle Produkte listen
    elif any(word in msg_lower for word in ['alle produkte', 'liste', 'verfügbar', 'welche produkte gibt']):
        return list_all_products.invoke({})
    
    # Top Potential
    elif any(word in msg_lower for word in ['top potential', 'top-potential', 'top-kunde', 'top-gewinn']):
        return top_potential_customer.invoke({"top_k": 3})
    
    # Default Antwort
    else:
        return f"""Ich bin Ihr Bank-Berater AI Assistant. Ich kann Ihnen bei folgenden Aufgaben helfen:

🎯 **Produktempfehlungen**: "Welche Produkte empfehlen Sie Kunde 17?"
📊 **Kunden-Snapshot**: "Zeigen Sie mir den Snapshot von Kunde 23"
📦 **Produkt-Details**: "Was kostet DepotPlus?" oder "Erklären Sie mir die GoldCard"
📋 **Produkt-Übersicht**: "Listen Sie alle verfügbaren Produkte auf"
💎 **Top-Potential-Kunde**: "Zeige mir den Kunden mit dem höchsten erwarteten Gewinn"

Aktueller Kunde: {customer_id if customer_id else 'Nicht ausgewählt'}

Wie kann ich Ihnen helfen?"""

def create_agent(api_key: str, customer_id: int = None):
    """
    Erstellt einen LangChain Agent mit OpenAI Function Calling
    
    Args:
        api_key: OpenAI API Key
        customer_id: Optionale Kunden-ID für Kontext
    
    Returns:
        AgentExecutor Instanz
    """
    
    # LLM mit Function Calling
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Tools
    tools = [recommend_products, get_customer_snapshot, explain_product, list_all_products, top_potential_customer]
    
    # System Prompt
    system_message = """Du bist ein hilfreicher Bank-Berater AI Assistant für die Bank-Adviser AI.
    
Du hast Zugriff auf folgende Tools:
- recommend_products: Empfiehlt Produkte für einen Kunden basierend auf ML-Modell
- get_customer_snapshot: Zeigt detaillierte Kundeninformationen
- explain_product: Erklärt ein spezifisches Bankprodukt
- list_all_products: Listet alle verfügbaren Produkte auf
- top_potential_customer: Liefert den Kunden mit dem höchsten erwarteten Gewinn basierend auf Produktempfehlungen

Nutze diese Tools, um präzise und hilfreiche Antworten zu geben.
Antworte immer auf Deutsch und sei freundlich und professionell.
"""
    
    if customer_id:
        customers_df, _, _, _ = get_data()
        customer = customers_df[customers_df['cust_id'] == customer_id]
        if not customer.empty:
            customer = customer.iloc[0]
            system_message += f"\n\nAktueller Kunde: {customer['first_name']} {customer['last_name']} (ID: {customer_id})"
    
    # Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    return agent_executor

def chat_llm(user_msg: str, api_key: str, customer_id: int = None, chat_history: List[Dict] = None) -> str:
    """
    Chat-Funktion für Streamlit Integration
    
    Args:
        user_msg: Nachricht des Benutzers
        api_key: OpenAI API Key
        customer_id: Optionale Kunden-ID für Kontext
        chat_history: Optionale Chat-Historie
    
    Returns:
        Antwort des Agenten
    """
    try:
        # Wenn kein API Key oder leer, nutze Mock
        if not api_key or api_key.strip() == "":
            return "🤖 **Mock-Modus** (Kein API Key)\n\n" + mock_chat_response(user_msg, customer_id)
        
        # Versuche mit OpenAI
        agent = create_agent(api_key, customer_id)
        
        # Konvertiere Chat-Historie wenn vorhanden
        if chat_history:
            # LangChain erwartet ein bestimmtes Format für die Historie
            pass  # Für diese Implementation nutzen wir erstmal keine Historie
        
        # Führe Agent aus
        result = agent.invoke({"input": user_msg})
        
        return result['output']
        
    except Exception as e:
        error_str = str(e)
        
        # Spezielle Behandlung für Quota-Fehler
        if "insufficient_quota" in error_str or "429" in error_str:
            return f"""❌ **OpenAI API Quota-Fehler**

Ihr OpenAI-Account hat das Limit erreicht oder es fehlt eine Zahlungsmethode.

**Lösungsmöglichkeiten:**
1. Überprüfen Sie Ihr OpenAI-Konto: https://platform.openai.com/usage
2. Fügen Sie eine Zahlungsmethode hinzu: https://platform.openai.com/account/billing
3. Warten Sie bis zum nächsten Abrechnungszeitraum

**Alternative: Mock-Modus aktivieren**
Löschen Sie einfach Ihren API Key in der Sidebar, dann nutzt das System den Mock-Modus.

---
**Mock-Antwort für Ihre Frage:**

""" + mock_chat_response(user_msg, customer_id)
        
        # Andere Fehler
        else:
            return f"""❌ **Fehler:** {error_str}

Bitte überprüfen Sie Ihren API Key oder nutzen Sie den Mock-Modus (API Key leer lassen).

---
**Mock-Antwort für Ihre Frage:**

""" + mock_chat_response(user_msg, customer_id)

if __name__ == "__main__":
    # Test des Agents
    print("🏦 Bank-Adviser AI - LLM Agent Test")
    print("=" * 50)
    
    # Beispiel-Anfragen
    test_queries = [
        "Welche Produkte würdest du Kunde 17 empfehlen?",
        "Zeig mir bitte den Snapshot von Kunde 23.",
        "Was kostet DepotPlus?",
        "Liste alle verfügbaren Produkte auf.",
        "Zeige mir den Kunden mit dem höchsten erwarteten Gewinn."
    ]
    
    # Test Mock-Modus
    print("\n🤖 Mock-Modus Test (ohne API Key):")
    for query in test_queries:
        print(f"\n❓ Frage: {query}")
        print(f"💬 Antwort: {chat_llm(query, api_key='', customer_id=1)}") 