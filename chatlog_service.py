#!/usr/bin/env python3
"""
Chatlog Service für Bank-Adviser AI
Speichert und verwaltet Chat-Verläufe für jeden Kunden
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

# Verzeichnis für Chatlogs
CHATLOG_DIR = "data/chatlogs"

def ensure_chatlog_dir():
    """Stelle sicher, dass das Chatlog-Verzeichnis existiert"""
    if not os.path.exists(CHATLOG_DIR):
        os.makedirs(CHATLOG_DIR)

def get_chatlog_filepath(customer_id: int) -> str:
    """Gibt den Dateipfad für Chatlogs eines Kunden zurück"""
    ensure_chatlog_dir()
    return os.path.join(CHATLOG_DIR, f"customer_{customer_id}_chatlogs.json")

def save_chatlog(customer_id: int, chat_history: List[Dict], metadata: Optional[Dict] = None):
    """
    Speichert einen Chat-Verlauf für einen Kunden
    
    Args:
        customer_id: ID des Kunden
        chat_history: Liste von Chat-Nachrichten mit 'role' und 'content'
        metadata: Optionale Metadaten (z.B. Zeitstempel, Session-Info)
    """
    filepath = get_chatlog_filepath(customer_id)
    
    # Lade existierende Chatlogs wenn vorhanden
    existing_logs = load_all_chatlogs(customer_id)
    
    # Erstelle neuen Chatlog-Eintrag
    new_log = {
        'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'timestamp': datetime.now().isoformat(),
        'customer_id': customer_id,
        'messages': chat_history,
        'message_count': len(chat_history),
        'metadata': metadata or {}
    }
    
    # Füge neuen Log hinzu
    existing_logs.append(new_log)
    
    # Speichere alle Logs
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(existing_logs, f, ensure_ascii=False, indent=2)

def load_all_chatlogs(customer_id: int) -> List[Dict]:
    """
    Lädt alle Chat-Verläufe für einen Kunden
    
    Args:
        customer_id: ID des Kunden
    
    Returns:
        Liste aller Chatlog-Einträge
    """
    filepath = get_chatlog_filepath(customer_id)
    
    if not os.path.exists(filepath):
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def load_latest_chatlog(customer_id: int) -> Optional[Dict]:
    """
    Lädt den neuesten Chat-Verlauf für einen Kunden
    
    Args:
        customer_id: ID des Kunden
    
    Returns:
        Der neueste Chatlog-Eintrag oder None
    """
    all_logs = load_all_chatlogs(customer_id)
    return all_logs[-1] if all_logs else None

def delete_chatlog(customer_id: int, session_id: str) -> bool:
    """
    Löscht einen spezifischen Chatlog
    
    Args:
        customer_id: ID des Kunden
        session_id: ID der Chat-Session
    
    Returns:
        True wenn erfolgreich gelöscht, False sonst
    """
    all_logs = load_all_chatlogs(customer_id)
    filtered_logs = [log for log in all_logs if log.get('session_id') != session_id]
    
    if len(filtered_logs) < len(all_logs):
        filepath = get_chatlog_filepath(customer_id)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(filtered_logs, f, ensure_ascii=False, indent=2)
        return True
    return False

def delete_all_chatlogs(customer_id: int) -> bool:
    """
    Löscht alle Chatlogs für einen Kunden
    
    Args:
        customer_id: ID des Kunden
    
    Returns:
        True wenn erfolgreich gelöscht
    """
    filepath = get_chatlog_filepath(customer_id)
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False

def get_chatlog_statistics(customer_id: int) -> Dict:
    """
    Gibt Statistiken über die Chatlogs eines Kunden zurück
    
    Args:
        customer_id: ID des Kunden
    
    Returns:
        Dictionary mit Statistiken
    """
    all_logs = load_all_chatlogs(customer_id)
    
    if not all_logs:
        return {
            'total_sessions': 0,
            'total_messages': 0,
            'avg_messages_per_session': 0,
            'first_chat': None,
            'last_chat': None
        }
    
    total_messages = sum(log.get('message_count', 0) for log in all_logs)
    
    return {
        'total_sessions': len(all_logs),
        'total_messages': total_messages,
        'avg_messages_per_session': total_messages / len(all_logs) if all_logs else 0,
        'first_chat': all_logs[0].get('timestamp'),
        'last_chat': all_logs[-1].get('timestamp')
    }

def search_chatlogs(customer_id: int, search_term: str) -> List[Dict]:
    """
    Durchsucht Chatlogs nach einem Begriff
    
    Args:
        customer_id: ID des Kunden
        search_term: Suchbegriff
    
    Returns:
        Liste von Chatlogs die den Begriff enthalten
    """
    all_logs = load_all_chatlogs(customer_id)
    search_term_lower = search_term.lower()
    
    matching_logs = []
    for log in all_logs:
        # Durchsuche alle Nachrichten
        for msg in log.get('messages', []):
            if search_term_lower in msg.get('content', '').lower():
                matching_logs.append(log)
                break  # Ein Treffer pro Log reicht
    
    return matching_logs

def export_chatlogs_to_csv(customer_id: int, filepath: str):
    """
    Exportiert alle Chatlogs eines Kunden in eine CSV-Datei
    
    Args:
        customer_id: ID des Kunden
        filepath: Pfad zur CSV-Datei
    """
    all_logs = load_all_chatlogs(customer_id)
    
    # Erstelle flache Struktur für CSV
    rows = []
    for log in all_logs:
        session_id = log.get('session_id')
        timestamp = log.get('timestamp')
        
        for msg in log.get('messages', []):
            rows.append({
                'customer_id': customer_id,
                'session_id': session_id,
                'timestamp': timestamp,
                'role': msg.get('role'),
                'content': msg.get('content')
            })
    
    # Speichere als CSV
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False, encoding='utf-8')

def get_all_customers_with_chatlogs() -> List[int]:
    """
    Gibt eine Liste aller Kunden-IDs zurück, die Chatlogs haben
    
    Returns:
        Liste von Kunden-IDs
    """
    ensure_chatlog_dir()
    customer_ids = []
    
    for filename in os.listdir(CHATLOG_DIR):
        if filename.startswith('customer_') and filename.endswith('_chatlogs.json'):
            try:
                # Extrahiere Kunden-ID aus Dateiname
                customer_id = int(filename.split('_')[1])
                customer_ids.append(customer_id)
            except:
                continue
    
    return sorted(customer_ids)

if __name__ == "__main__":
    # Test des Chatlog Service
    print("🏦 Bank-Adviser AI - Chatlog Service Test")
    print("=" * 50)
    
    # Test-Chatlog
    test_chat = [
        {'role': 'user', 'content': 'Welche Produkte empfehlen Sie mir?'},
        {'role': 'assistant', 'content': 'Basierend auf Ihrem Profil empfehle ich Ihnen DepotBasic.'},
        {'role': 'user', 'content': 'Was kostet das?'},
        {'role': 'assistant', 'content': 'DepotBasic kostet 120€ pro Jahr.'}
    ]
    
    # Speichere Test-Chatlog
    print("\n💾 Speichere Test-Chatlog für Kunde 1...")
    save_chatlog(1, test_chat, {'test': True})
    
    # Lade Chatlogs
    print("\n📂 Lade alle Chatlogs für Kunde 1...")
    logs = load_all_chatlogs(1)
    print(f"Gefunden: {len(logs)} Chatlog(s)")
    
    # Statistiken
    print("\n📊 Chatlog-Statistiken für Kunde 1:")
    stats = get_chatlog_statistics(1)
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    # Suche
    print("\n🔍 Suche nach 'DepotBasic'...")
    results = search_chatlogs(1, 'DepotBasic')
    print(f"Gefunden in {len(results)} Chatlog(s)")
    
    print("\n✅ Chatlog Service bereit!") 