import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'radiology.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create patients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create scans table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            scan_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            original_image_path TEXT NOT NULL,
            heatmap_image_path TEXT NOT NULL,
            original_filename TEXT,
            view_position TEXT DEFAULT 'PA',
            clinical_indication TEXT,
            primary_finding TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL,
            findings_json TEXT NOT NULL,
            report_text TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_patient_and_scan(patient_info, scan_results):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Insert or update patient
    cursor.execute('''
        INSERT INTO patients (patient_id, full_name, age, gender)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(patient_id) DO UPDATE SET
            full_name = excluded.full_name,
            age = excluded.age,
            gender = excluded.gender
    ''', (
        patient_info.get('patient_id', 'ANON-' + datetime.now().strftime('%Y%m%d%H%M')),
        patient_info.get('full_name', 'Anonymous Patient'),
        int(patient_info.get('age', 45)),
        patient_info.get('gender', 'Unknown')
    ))
    
    # Insert scan record
    cursor.execute('''
        INSERT INTO scans (
            patient_id, original_image_path, heatmap_image_path, original_filename,
            view_position, clinical_indication, primary_finding, confidence,
            severity, findings_json, report_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        patient_info.get('patient_id'),
        scan_results.get('original_image_path'),
        scan_results.get('heatmap_image_path'),
        scan_results.get('original_filename'),
        patient_info.get('view_position', 'PA'),
        patient_info.get('clinical_indication', 'Routine screening'),
        scan_results.get('primary_finding'),
        float(scan_results.get('confidence', 0.0)),
        scan_results.get('severity', 'Routine'),
        json.dumps(scan_results.get('all_findings', {})),
        scan_results.get('report_text', '')
    ))
    
    scan_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return scan_id

def get_all_scans(limit=100):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT s.*, p.full_name, p.age, p.gender
        FROM scans s
        LEFT JOIN patients p ON s.patient_id = p.patient_id
        ORDER BY s.id DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    scans = [dict(row) for row in rows]
    for s in scans:
        if 'findings_json' in s and s['findings_json']:
            try:
                s['findings'] = json.loads(s['findings_json'])
            except Exception:
                s['findings'] = {}
    conn.close()
    return scans

def get_scan_by_id(scan_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT s.*, p.full_name, p.age, p.gender
        FROM scans s
        LEFT JOIN patients p ON s.patient_id = p.patient_id
        WHERE s.id = ?
    ''', (scan_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        data = dict(row)
        try:
            data['findings'] = json.loads(data['findings_json'])
        except Exception:
            data['findings'] = {}
        return data
    return None

def delete_scan(scan_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM scans WHERE id = ?', (scan_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def get_analytics_summary():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) as total FROM scans')
    total_scans = cursor.fetchone()['total']
    
    cursor.execute('SELECT COUNT(*) as abnormal FROM scans WHERE primary_finding != "No Acute Finding (Normal)"')
    abnormal_scans = cursor.fetchone()['abnormal']
    
    cursor.execute('SELECT COUNT(DISTINCT patient_id) as total_patients FROM patients')
    total_patients = cursor.fetchone()['total_patients']
    
    cursor.execute('''
        SELECT primary_finding, COUNT(*) as count 
        FROM scans 
        GROUP BY primary_finding 
        ORDER BY count DESC 
        LIMIT 5
    ''')
    top_findings = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return {
        'total_scans': total_scans,
        'abnormal_scans': abnormal_scans,
        'normal_scans': total_scans - abnormal_scans,
        'total_patients': total_patients,
        'top_findings': top_findings
    }
