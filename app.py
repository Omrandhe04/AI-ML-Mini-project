import os
import shutil
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename

import database
import ai_engine

app = Flask(__name__)
app.config['SECRET_KEY'] = 'chest-xray-ai-cad-secure-key-2026'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
HEATMAP_FOLDER = os.path.join(BASE_DIR, 'static', 'heatmaps')
SAMPLE_FOLDER = os.path.join(BASE_DIR, 'static', 'sample_images')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_FOLDER, exist_ok=True)

# Initialize database
database.init_db()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp', 'dcm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    # Read patient metadata
    patient_id = request.form.get('patient_id', '').strip()
    full_name = request.form.get('full_name', '').strip()
    age = request.form.get('age', '45')
    gender = request.form.get('gender', 'M')
    view_position = request.form.get('view_position', 'PA')
    clinical_indication = request.form.get('clinical_indication', 'Routine CXR screening').strip()
    sample_select = request.form.get('sample_select', '').strip()

    if not patient_id:
        patient_id = f"PAT-{uuid.uuid4().hex[:6].upper()}"
    if not full_name:
        full_name = "Anonymous Patient"

    patient_info = {
        'patient_id': patient_id,
        'full_name': full_name,
        'age': int(age) if str(age).isdigit() else 45,
        'gender': gender,
        'view_position': view_position,
        'clinical_indication': clinical_indication
    }

    saved_filename = ""
    saved_filepath = ""

    # Check if a sample was selected or an image was uploaded
    if 'image' in request.files and request.files['image'].filename:
        file = request.files['image']
        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file format. Please upload PNG, JPG, or JPEG.'}), 400
        
        ext = file.filename.rsplit('.', 1)[1].lower()
        saved_filename = f"cxr_{uuid.uuid4().hex[:8]}.{ext}"
        saved_filepath = os.path.join(UPLOAD_FOLDER, saved_filename)
        file.save(saved_filepath)
        original_disp_name = file.filename
    elif sample_select:
        # User selected one of the built-in sample images
        sample_path = os.path.join(SAMPLE_FOLDER, f"{sample_select}.jpg")
        if not os.path.exists(sample_path):
            return jsonify({'error': f"Sample '{sample_select}' not found."}), 400
        
        saved_filename = f"cxr_{sample_select}_{uuid.uuid4().hex[:6]}.jpg"
        saved_filepath = os.path.join(UPLOAD_FOLDER, saved_filename)
        shutil.copyfile(sample_path, saved_filepath)
        original_disp_name = f"{sample_select.capitalize()} Chest X-Ray"
    else:
        return jsonify({'error': 'Please provide an image file or choose a sample scan.'}), 400

    # Execute AI Scan and Heatmap Engine
    try:
        analysis_res = ai_engine.analyze_chest_xray(saved_filepath, patient_info, HEATMAP_FOLDER)
    except Exception as e:
        return jsonify({'error': f"Error during AI analysis: {str(e)}"}), 500

    original_image_web_path = f"/static/uploads/{saved_filename}"

    # Package scan data
    scan_results = {
        'original_image_path': original_image_web_path,
        'heatmap_image_path': analysis_res['heatmap_image_path'],
        'original_filename': original_disp_name,
        'primary_finding': analysis_res['primary_finding'],
        'confidence': analysis_res['confidence'],
        'severity': analysis_res['severity'],
        'all_findings': analysis_res['all_findings'],
        'report_text': analysis_res['report_text']
    }

    # Save to SQLite Database
    scan_id = database.save_patient_and_scan(patient_info, scan_results)

    return jsonify({
        'success': True,
        'scan_id': scan_id,
        'patient_info': patient_info,
        'analysis': scan_results
    })

@app.route('/api/scans', methods=['GET'])
def get_scans():
    scans = database.get_all_scans()
    return jsonify({'scans': scans})

@app.route('/api/scan/<int:scan_id>', methods=['GET'])
def get_scan(scan_id):
    scan = database.get_scan_by_id(scan_id)
    if not scan:
        return jsonify({'error': 'Scan record not found'}), 404
    return jsonify({'scan': scan})

@app.route('/api/delete/<int:scan_id>', methods=['POST', 'DELETE'])
def delete_scan(scan_id):
    success = database.delete_scan(scan_id)
    return jsonify({'success': success})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    stats = database.get_analytics_summary()
    return jsonify(stats)

@app.route('/report/<int:scan_id>')
def view_printable_report(scan_id):
    scan = database.get_scan_by_id(scan_id)
    if not scan:
        return "Scan not found", 404
    return render_template('report_print.html', scan=scan)

if __name__ == '__main__':
    print("=================================================================")
    print("  RadVision AI - Chest X-Ray CAD & Heatmap Localization System")
    print("  Server launching on: http://127.0.0.1:5000")
    print("  Database: SQLite (radiology.db)")
    print("=================================================================")
    app.run(host='0.0.0.0', port=5000, debug=True)
