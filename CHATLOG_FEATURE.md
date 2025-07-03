# Chat-Historie Feature

## Übersicht

Das neue Chat-Historie Feature ermöglicht es, Chatverläufe für jeden Kunden zu speichern und anzuzeigen. Die Chatlogs werden dauerhaft im Dateisystem gespeichert und können durchsucht, angezeigt und exportiert werden.

## Features

### 1. **Automatisches Speichern**
- Chats können automatisch gespeichert werden, wenn der Kunde gewechselt wird
- Option kann in der Sidebar aktiviert/deaktiviert werden

### 2. **Manuelles Speichern**
- "Speichern" Button im Chat-Tab
- Speichert den aktuellen Chatverlauf für den ausgewählten Kunden

### 3. **Chat-Historie Tab**
- Neuer Tab "📜 Chat-Historie" in der Hauptnavigation
- Zeigt alle gespeicherten Chats für den aktuellen Kunden

### 4. **Such-Funktionalität**
- Durchsuchen der Chatlogs nach Stichwörtern
- Filtert die Anzeige basierend auf dem Suchbegriff

### 5. **Statistiken**
- Anzahl gespeicherter Chats
- Gesamtzahl der Nachrichten
- Durchschnittliche Nachrichten pro Chat
- Zeitstempel des ersten und letzten Chats

### 6. **Export-Optionen**
- Export aller Chatlogs eines Kunden als CSV
- Format: customer_id, session_id, timestamp, role, content

### 7. **Lösch-Funktionen**
- Einzelne Chats können gelöscht werden
- Alle Chats eines Kunden können auf einmal gelöscht werden

## Technische Details

### Speicherort
- Chatlogs werden in `data/chatlogs/` gespeichert
- Format: `customer_{ID}_chatlogs.json`

### Datenstruktur
```json
{
  "session_id": "20240101_120000",
  "timestamp": "2024-01-01T12:00:00",
  "customer_id": 1,
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "message_count": 4,
  "metadata": {
    "saved_at": "2024-01-01T12:05:00",
    "api_mode": "OpenAI",
    "auto_saved": false
  }
}
```

### Neue Module
- `chatlog_service.py`: Service für Chatlog-Management
  - `save_chatlog()`: Speichert einen Chatverlauf
  - `load_all_chatlogs()`: Lädt alle Chatlogs eines Kunden
  - `search_chatlogs()`: Durchsucht Chatlogs
  - `get_chatlog_statistics()`: Berechnet Statistiken
  - `export_chatlogs_to_csv()`: CSV-Export

## Verwendung

### Für Benutzer
1. Führen Sie einen Chat im "Chat mit AI" Tab
2. Klicken Sie auf "💾 Speichern" oder aktivieren Sie "Chats automatisch speichern"
3. Wechseln Sie zum "📜 Chat-Historie" Tab um gespeicherte Chats anzuzeigen
4. Nutzen Sie die Suchfunktion um spezifische Unterhaltungen zu finden
5. Exportieren Sie bei Bedarf die Chatlogs als CSV

### Für Entwickler
```python
import chatlog_service

# Chat speichern
chatlog_service.save_chatlog(
    customer_id=1,
    chat_history=[
        {'role': 'user', 'content': 'Hallo'},
        {'role': 'assistant', 'content': 'Wie kann ich helfen?'}
    ],
    metadata={'api_mode': 'OpenAI'}
)

# Chatlogs laden
logs = chatlog_service.load_all_chatlogs(customer_id=1)

# Suchen
results = chatlog_service.search_chatlogs(customer_id=1, search_term="Depot")
```

## Datenschutz
- Alle Chatlogs werden lokal gespeichert
- Keine Cloud-Synchronisation
- Löschfunktionen ermöglichen vollständige Entfernung 