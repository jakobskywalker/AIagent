# 🏦 Bank-Adviser AI - MVP

Ein KI-gestütztes Empfehlungssystem für Bankberater mit lokalem Machine Learning und Chat-Interface.

## 🚀 Quick Start

```bash
# 1. Dependencies installieren
pip install -r requirements.txt

# 2. Daten generieren
python data/make_kunde.py
python data/make_produkt.py
python data/make_kunde_produkt.py
python data/make_sales_history.py

# 3. Features erstellen
python -c "exec(open('notebooks/feature_build.ipynb').read())"
# oder nutzen Sie Jupyter Notebook

# 4. ML-Modell trainieren
python model_train.py

# 5. Streamlit App starten
streamlit run app.py
```

Öffnen Sie http://localhost:8501 in Ihrem Browser.

## 📁 Projektstruktur

```
MVP/
├── data/                      # Daten und Modelle
│   ├── customers.csv         # 50 synthetische Kunden
│   ├── products.csv          # 6 Bankprodukte
│   ├── ownership.csv         # Kunde-Produkt Zuordnungen
│   ├── sales_history.csv     # Transaktionshistorie
│   ├── features.parquet      # ML-Features
│   ├── model.pkl            # Trainiertes Modell
│   └── make_*.py            # Daten-Generierungs-Skripte
├── notebooks/
│   └── feature_build.ipynb   # Feature Engineering
├── app.py                    # Streamlit Web-App
├── agent_service.py          # Empfehlungs-Service
├── model_train.py           # ML-Training
└── requirements.txt         # Python Dependencies
```

## 🎯 Features

### 1. **Produktempfehlungen** 
- ML-basierte Vorhersagen (LogisticRegression)
- Confidence Scores für jedes Produkt
- Personalisierte Begründungen
- Top-K Empfehlungen

### 2. **Kunden-Snapshot**
- Aktuelle Produktübersicht
- Gesamtumsatz und Transaktionen
- Jährliche Gebühren
- Produktnutzung

### 3. **Produkt-Erklärungen**
- Detaillierte Produktbeschreibungen
- Risiko-Klassifizierung
- Kategorie-Informationen
- Gebührenübersicht

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML**: scikit-learn (LogisticRegression)
- **Daten**: pandas, pyarrow
- **AI**: OpenAI GPT-3.5 (optional)
- **Deployment**: Docker-ready

## 📊 Datenmodell

### customers.csv
- `cust_id`: Kunden-ID (1-50)
- `first_name`, `last_name`: Name
- `location`: Stadt
- `revenue`: Einkommen (20k-250k €)
- `credit_score`: Bonität (300-850)
- `age_bucket`: Altersgruppe

### products.csv
- `prod_id`: Produkt-ID (101-106)
- `name`: Produktname
- `category`: Giro/Depot/Kreditkarte/Versicherung
- `risk_class`: niedrig/mittel/hoch
- `price`: Jahresgebühr
- `short_desc`: Kurzbeschreibung

### ownership.csv
- `cust_id`: Kunden-ID
- `prod_id`: Produkt-ID
- `since_date`: Besitzdatum

## 🤖 ML-Modell

- **Algorithmus**: LogisticRegression
- **Target**: Produkt 102 (DepotBasic)
- **Features**: Alter, Einkommen, Credit Score, andere Produkte
- **Performance**: ROC-AUC 0.81

## 🔒 Datenschutz

- ✅ Alle Daten bleiben lokal
- ✅ Kein PII verlässt das System
- ✅ EU-konform
- ✅ Optional: OpenAI API für Erklärungen

## 🐳 Docker Deployment

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 📝 Verwendung

1. **Kunde auswählen**: Sidebar → Kundennummer (1-50)
2. **Empfehlungen**: Tab 1 → "Empfehlung anfordern"
3. **Snapshot**: Tab 2 → "Snapshot anzeigen"
4. **Produkte**: Tab 3 → Produkt auswählen → "Erklären"

## 🔧 Konfiguration

### OpenAI Integration (optional)
```bash
export OPENAI_API_KEY="sk-..."
```

Ohne API-Key nutzt das System regelbasierte Erklärungen.

## 📈 Erweiterungsmöglichkeiten

- [ ] Weitere ML-Modelle für alle Produkte
- [ ] Echtzeit-Datenanbindung
- [ ] A/B Testing Framework
- [ ] Multi-Language Support
- [ ] Advanced Analytics Dashboard

## 👥 Team

Entwickelt als MVP für Bank-Adviser AI System.

---

**Hinweis**: Dies ist ein MVP mit synthetischen Daten zu Demonstrationszwecken. 