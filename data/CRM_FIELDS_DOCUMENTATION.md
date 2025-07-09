# CRM-Felder Dokumentation

## Übersicht
Diese Dokumentation beschreibt die neu hinzugefügten CRM-relevanten Felder in den Datentabellen.

## Neue Felder

### sales_history.csv
Die Verkaufshistorie wurde um folgende Felder erweitert:

| Feldname | Datentyp | Beschreibung | Beispielwerte |
|----------|----------|--------------|---------------|
| **kundenbetreuer_id** | Integer/NULL | ID des zuständigen Kundenbetreuers für diese Transaktion | 101, 102, NULL |
| **entscheidung** | String/NULL | Status der Verkaufsentscheidung | 'abgeschlossen', 'abgelehnt', 'vertagt', 'in_bearbeitung' |
| **begruendung** | Text/NULL | Freitext-Begründung für die Entscheidung oder Notizen | 'Kunde hat sich für günstigeres Angebot entschieden', 'Beratung erfolgreich' |

### ownership.csv
Die Produktzugehörigkeit wurde um folgendes Feld erweitert:

| Feldname | Datentyp | Beschreibung | Beispielwerte |
|----------|----------|--------------|---------------|
| **kundenbetreuer_id** | Integer/NULL | ID des Kundenbetreuers, der das Produkt verkauft hat | 101, 102, NULL |

## Verwendungszweck

Diese Felder ermöglichen:
- **Vertriebsanalyse**: Welcher Kundenbetreuer hat welche Erfolgsquote?
- **Entscheidungstracking**: Nachvollziehbarkeit von Verkaufsentscheidungen
- **Performance-Messung**: KPIs für einzelne Kundenbetreuer
- **Prozessoptimierung**: Analyse von Ablehnungsgründen

## Implementierung

Alle Felder wurden initial mit NULL-Werten gefüllt und können schrittweise mit Daten angereichert werden.

## Beispiel-Queries

```sql
-- Erfolgsquote pro Kundenbetreuer
SELECT 
    kundenbetreuer_id,
    COUNT(CASE WHEN entscheidung = 'abgeschlossen' THEN 1 END) as erfolge,
    COUNT(*) as gesamt,
    ROUND(COUNT(CASE WHEN entscheidung = 'abgeschlossen' THEN 1 END) * 100.0 / COUNT(*), 2) as erfolgsquote
FROM sales_history
WHERE kundenbetreuer_id IS NOT NULL
GROUP BY kundenbetreuer_id;

-- Häufigste Ablehnungsgründe
SELECT 
    begruendung,
    COUNT(*) as anzahl
FROM sales_history
WHERE entscheidung = 'abgelehnt'
    AND begruendung IS NOT NULL
GROUP BY begruendung
ORDER BY anzahl DESC;
``` 