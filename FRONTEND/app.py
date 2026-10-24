from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import joblib
import numpy as np
import pandas as pd
import sqlite3
import os
from datetime import datetime
import random
import uuid
from collections import OrderedDict
import ct_model
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024      # CT uploads up to 8 MB
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}
app.secret_key = os.environ.get('SECRET_KEY', 'pulmocare_ai_secret_key_2026_x89f')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'users.db')
MODEL_PATH = os.path.join(BASE_DIR, 'perceptron.pkl')

# ===========================================================================
# Load Pre-trained Model
# ===========================================================================
try:
    model = joblib.load(MODEL_PATH)
    print(f"[*] Perceptron model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"[!] Warning: Could not load model from {MODEL_PATH}: {e}")
    model = None

# Model features in exact training order
FEATURE_ORDER = [
    'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
    'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
    'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
    'SWALLOWING DIFFICULTY', 'CHEST PAIN'
]

# Field mappings: (model_key, [possible_form_names], human_label)
FEATURE_MAPPINGS = [
    ('GENDER', ['gender', 'GENDER'], 'Biological Sex'),
    ('AGE', ['age', 'AGE'], 'Age'),
    ('SMOKING', ['smoking', 'SMOKING'], 'Tobacco Smoking'),
    ('YELLOW_FINGERS', ['yellow_fingers', 'YELLOW_FINGERS'], 'Yellow / Nicotine Fingers'),
    ('ANXIETY', ['anxiety', 'ANXIETY'], 'Chronic Anxiety'),
    ('PEER_PRESSURE', ['peer_pressure', 'PEER_PRESSURE'], 'Peer / Secondhand Smoke Exposure'),
    ('CHRONIC DISEASE', ['chronic_disease', 'CHRONIC_DISEASE', 'CHRONIC DISEASE'], 'Chronic Disease'),
    ('FATIGUE ', ['fatigue', 'FATIGUE', 'FATIGUE '], 'Fatigue / Lethargy'),
    ('ALLERGY ', ['allergy', 'ALLERGY', 'ALLERGY '], 'Allergy History'),
    ('WHEEZING', ['wheezing', 'WHEEZING'], 'Wheezing Sounds'),
    ('ALCOHOL CONSUMING', ['alcohol_consuming', 'ALCOHOL_CONSUMING', 'ALCOHOL CONSUMING'], 'Alcohol Consumption'),
    ('COUGHING', ['coughing', 'COUGHING'], 'Persistent Coughing'),
    ('SHORTNESS OF BREATH', ['shortness_of_breath', 'SHORTNESS_OF_BREATH', 'SHORTNESS OF BREATH'], 'Shortness of Breath'),
    ('SWALLOWING DIFFICULTY', ['swallowing_difficulty', 'SWALLOWING_DIFFICULTY', 'SWALLOWING DIFFICULTY'], 'Swallowing Difficulty'),
    ('CHEST PAIN', ['chest_pain', 'CHEST_PAIN', 'CHEST PAIN'], 'Chest Discomfort / Pain')
]

# ===========================================================================
# Database Initialization & Management
# ===========================================================================
def init_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT,
                    password TEXT NOT NULL
                )
            ''')
            # Check if email column exists
            c.execute('PRAGMA table_info(users)')
            cols = [row[1] for row in c.fetchall()]
            if 'email' not in cols:
                c.execute('ALTER TABLE users ADD COLUMN email TEXT')

            # Seed demo doctor user if not present
            c.execute('SELECT id FROM users WHERE username = ?', ('doctor',))
            if not c.fetchone():
                hashed_pw = generate_password_hash('doctor123')
                c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                          ('doctor', 'doctor@pulmocare.org', hashed_pw))
            c.execute('''
                CREATE TABLE IF NOT EXISTS assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    report_id TEXT,
                    age INTEGER,
                    gender TEXT,
                    symptom_high INTEGER,
                    symptom_count INTEGER,
                    ct_label TEXT,
                    ct_confidence REAL,
                    ct_model TEXT,
                    final_level TEXT,
                    final_title TEXT
                )
            ''')
            c.execute('PRAGMA table_info(assessments)')
            acols = [row[1] for row in c.fetchall()]
            if 'patient_name' not in acols:
                c.execute('ALTER TABLE assessments ADD COLUMN patient_name TEXT')
            conn.commit()
    except Exception as e:
        print(f"[!] Database init error: {e}")

# Run DB init on launch
init_db()

# ===========================================================================
# Helper: Extract Form Features
# ===========================================================================
def extract_features(form_data):
    features_dict = {}
    breakdown = []
    
    for model_key, form_keys, label in FEATURE_MAPPINGS:
        val = None
        for k in form_keys:
            if k in form_data:
                val = form_data.get(k)
                break
        
        # Default fallback
        if val is None or val == '':
            float_val = 0.0
        else:
            try:
                float_val = float(val)
            except (ValueError, TypeError):
                float_val = 0.0
                
        features_dict[model_key] = float_val
        
        # Binary symptoms breakdown (skip gender & age for symptom list)
        if model_key not in ['GENDER', 'AGE']:
            breakdown.append({
                'key': model_key,
                'label': label,
                'present': bool(float_val >= 1.0)
            })
            
    # Form a DataFrame with precise columns to match scikit-learn model
    df_input = pd.DataFrame([[features_dict[col] for col in FEATURE_ORDER]], columns=FEATURE_ORDER)
    return df_input, features_dict, breakdown

# ===========================================================================
# Routes
# ===========================================================================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            if model is None:
                flash("Model file perceptron.pkl is not loaded.", "danger")
                return redirect(url_for('home'))

            # Extract features and symptom breakdown
            df_input, features_dict, symptom_breakdown = extract_features(request.form)

            # Predict using pre-trained Perceptron
            prediction_val = int(model.predict(df_input)[0])
            is_detected = (prediction_val == 1)

            # Positive symptoms count (excluding Demographics)
            positive_count = sum(1 for item in symptom_breakdown if item['present'])

            # Report reference metadata
            report_id = f"LENS-{random.randint(1000, 9999)}"
            timestamp = datetime.now().strftime("%b %d, %Y - %I:%M %p")

            patient_data = {
                'gender': str(int(features_dict.get('GENDER', 1))),
                'age': int(features_dict.get('AGE', 45)),
            }

            patient_name = request.form.get('patient_name', '').strip()[:80]
            session['symptom'] = {'patient_name': patient_name, 'high': bool(is_detected), 'age': patient_data['age'], 'gender': patient_data['gender'],
                                  'positive_count': positive_count, 'report_id': report_id, 'timestamp': timestamp,
                                  'items': [i['label'] for i in symptom_breakdown if i['present']]}
            if session.get('flow'):
                return redirect(url_for('ct'))

            result_text = "Lung Disease Indication Detected" if is_detected else "No Lung Disease Indication Detected"

            return render_template(
                'result.html',
                prediction=result_text,
                is_detected=is_detected,
                patient_data=patient_data,
                symptom_breakdown=symptom_breakdown,
                positive_count=positive_count,
                report_id=report_id,
                timestamp=timestamp
            )

        except Exception as e:
            flash(f"Error encountered during clinical prediction: {str(e)}", "danger")
            return redirect(url_for('predict'))

    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for asynchronous or mobile screening inference"""
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500

        data = request.get_json() or request.form
        df_input, features_dict, symptom_breakdown = extract_features(data)

        prediction_val = int(model.predict(df_input)[0])
        is_detected = (prediction_val == 1)
        positive_count = sum(1 for item in symptom_breakdown if item['present'])

        return jsonify({
            'success': True,
            'prediction': prediction_val,
            'is_detected': is_detected,
            'risk_status': 'High Risk' if is_detected else 'Low Risk',
            'positive_symptoms_count': positive_count,
            'evaluated_features': features_dict
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ===========================================================================
# CT Scan Analysis (deep learning + Grad-CAM) and Combined Report
# ===========================================================================
CT_CACHE = OrderedDict()          # in-memory only: nothing is written to disk
CT_CACHE_MAX = 50

def _cache_put(result):
    token = uuid.uuid4().hex
    CT_CACHE[token] = result
    while len(CT_CACHE) > CT_CACHE_MAX:
        CT_CACHE.popitem(last=False)
    return token

def _current_ct():
    return CT_CACHE.get(session.get('ct_token', ''))

def combine_assessment(symptom_high, ct_label):
    """CT-led decision support. The CT sets the level; the questionnaire can only raise it by one step, never lower it.
    Simple transparent rule - NOT clinically validated.
    symptom_high: True / False / None (not done);  ct_label: 'Normal' / 'Benign' / 'Malignant' / None (not done)."""
    def out(level, title, why, nxt, complete=True):
        tone = {'low': 'success', 'moderate': 'warning', 'high': 'danger', 'none': 'info'}[level]
        return {'level': level, 'tone': tone, 'title': title, 'why': why, 'next': nxt, 'complete': complete}
    if symptom_high is None and ct_label is None:
        return out('none', 'No assessment yet', 'Neither check has been completed.',
                   'Start the full assessment: questionnaire first, then the CT scan.', False)
    if ct_label is None:      # questionnaire only -> provisional
        if symptom_high:
            return out('moderate', 'Provisional: CT scan needed',
                       'The questionnaire shows higher risk, but it cannot see inside the lungs and early lung cancer often causes no symptoms.',
                       'Add a CT scan analysis and discuss your risk factors with a doctor.', False)
        return out('low', 'Provisional: CT scan still advised',
                   'The questionnaire shows lower risk, but a low-risk questionnaire does not rule out lung cancer (early disease often has no symptoms).',
                   'Add a CT scan analysis for a more complete picture.', False)
    if symptom_high is None:  # CT only -> provisional
        lv = {'Normal': 'low', 'Benign': 'moderate', 'Malignant': 'high'}[ct_label]
        return out(lv, 'Provisional: questionnaire not done', f'Only the CT result is available ({ct_label}).',
                   'Complete the symptom questionnaire for the full assessment.', False)
    if ct_label == 'Malignant':
        why = ('Both checks point the same way: higher symptom risk and a CT pattern consistent with a malignant case.' if symptom_high else
               'The two checks differ: few symptoms were reported, but the CT shows a pattern consistent with a malignant case. Early lung cancer often causes no symptoms, so the CT result takes priority.')
        return out('high', 'Urgent specialist referral', why,
                   'See a pulmonologist or oncologist promptly for confirmatory evaluation (radiologist review, further imaging, biopsy if advised).')
    if ct_label == 'Benign':
        if symptom_high:
            return out('high', 'Doctor review soon', 'A benign-pattern CT finding together with higher symptom risk should be reviewed soon.',
                       'Book a doctor review soon and ask about a follow-up scan.')
        return out('moderate', 'Follow-up scan advised', 'The CT shows a benign-pattern finding and the questionnaire shows lower risk.',
                   'Ask your doctor about a follow-up scan to monitor the finding.')
    # CT Normal
    if symptom_high:
        return out('moderate', 'Doctor review recommended',
                   'The two checks differ: the questionnaire shows higher risk, but the CT slice looks normal. One slice does not cover the whole lung, and the risk factors remain.',
                   'Discuss your risk factors with a doctor and ask whether a full CT study or low-dose CT screening is appropriate.')
    return out('low', 'Routine checkup', 'Both checks agree: no warning signs in the questionnaire and a normal CT slice.',
               'Continue healthy habits (no smoking, avoid secondhand smoke) and attend routine medical checkups.')

RULE_ROWS = [('Lower risk', 'Normal', 'LOW'), ('Higher risk', 'Normal', 'MODERATE'), ('Lower risk', 'Benign', 'MODERATE'),
             ('Higher risk', 'Benign', 'HIGH'), ('Any', 'Malignant', 'HIGH')]

@app.errorhandler(413)
def too_large(_e):
    flash("The file is too large. Please upload an image smaller than 8 MB.", "danger")
    return redirect(url_for('ct'))

@app.route('/ct', methods=['GET', 'POST'])
def ct():
    ready, message = ct_model.status()
    if request.method == 'POST':
        f = request.files.get('ct_image')
        if not f or not f.filename:
            flash("Please choose a CT image to upload.", "warning")
            return redirect(url_for('ct'))
        ext = f.filename.rsplit('.', 1)[-1].lower() if '.' in f.filename else ''
        if ext not in ALLOWED_EXT:
            flash("Unsupported file type. Please upload a PNG or JPG image.", "warning")
            return redirect(url_for('ct'))
        try:
            result = ct_model.predict(f.read())
        except ct_model.ModelUnavailable as e:
            flash(str(e), "danger")
            return redirect(url_for('ct'))
        except ct_model.InvalidImage as e:
            flash(str(e), "warning")
            return redirect(url_for('ct'))
        except Exception as e:
            flash(f"CT analysis failed: {e}", "danger")
            return redirect(url_for('ct'))
        result['report_id'] = f"LENS-CT-{random.randint(1000, 9999)}"
        result['timestamp'] = datetime.now().strftime("%b %d, %Y - %I:%M %p")
        session['ct_token'] = _cache_put(result)
        session['ct_summary'] = {'label': result['label'], 'confidence': result['confidence']}
        if session.get('flow'):
            return redirect(url_for('combined'))
        return redirect(url_for('ct_result'))
    return render_template('ct.html', model_ready=ready, model_message=message)

@app.route('/ct/result')
def ct_result():
    result = _current_ct()
    if not result:
        flash("No CT analysis found. Please upload a scan first.", "info")
        return redirect(url_for('ct'))
    return render_template('ct_result.html', r=result, has_symptom=('symptom' in session))

@app.context_processor
def inject_flow():
    """Shows the 3-step progress bar while a guided assessment is running."""
    if not session.get('flow'):
        return {'flow': None}
    step = {'predict': 1, 'ct': 2, 'ct_result': 2, 'combined': 3}.get(request.endpoint)
    return {'flow': {'step': step} if step else None}

@app.route('/assess')
def assess_start():
    """Start a new guided assessment: questionnaire -> CT scan -> final report."""
    for k in ('symptom', 'ct_token', 'ct_summary', 'saved_key', 'saved_id'):
        session.pop(k, None)
    session['flow'] = True
    return redirect(url_for('predict'))

def _save_assessment(symptom, ct_res, assessment):
    """Save a finished assessment for the signed-in user (no images are stored)."""
    key = f"{symptom['report_id']}|{session.get('ct_token')}"
    if session.get('saved_key') == key:
        return session.get('saved_id')
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute('INSERT INTO assessments (user_id, created_at, report_id, age, gender, symptom_high, symptom_count, ct_label, ct_confidence, ct_model, final_level, final_title, patient_name) '
                    'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (session['user_id'], datetime.now().strftime("%Y-%m-%d %H:%M"), symptom['report_id'], symptom['age'], symptom['gender'],
                     int(symptom['high']), symptom['positive_count'], ct_res['label'], float(ct_res['confidence']), ct_res.get('arch'),
                     assessment['level'].upper(), assessment['title'], symptom.get('patient_name') or 'Unnamed'))
        conn.commit()
        session['saved_key'], session['saved_id'] = key, cur.lastrowid
        return cur.lastrowid

@app.route('/combined')
def combined():
    symptom = session.get('symptom')
    ct_res = _current_ct()
    assessment = combine_assessment(symptom['high'] if symptom else None, ct_res['label'] if ct_res else None)
    saved_id = None
    if symptom and ct_res and session.get('user_id'):
        try:
            saved_id = _save_assessment(symptom, ct_res, assessment)
        except Exception as e:
            flash(f"Could not save the assessment: {e}", "warning")
    cur = (('Higher risk' if symptom and symptom['high'] else 'Lower risk'), ct_res['label'] if ct_res else None)
    html = render_template('combined.html', symptom=symptom, ct=ct_res, assessment=assessment, saved_id=saved_id,
                           signed_in=bool(session.get('user_id')), rule_rows=RULE_ROWS, current=cur,
                           timestamp=datetime.now().strftime("%b %d, %Y - %I:%M %p"))
    session['flow'] = False        # guided flow is finished
    return html

@app.route('/history')
def history():
    if not session.get('user_id'):
        flash("Please sign in to see your saved assessments.", "info")
        return redirect(url_for('login'))
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        q = request.args.get('q', '').strip()
        if q:
            rows = conn.execute('SELECT * FROM assessments WHERE user_id = ? AND patient_name LIKE ? ORDER BY id DESC',
                                (session['user_id'], f'%{q}%')).fetchall()
        else:
            rows = conn.execute('SELECT * FROM assessments WHERE user_id = ? ORDER BY id DESC', (session['user_id'],)).fetchall()
    return render_template('history.html', rows=rows, q=q)

@app.route('/history/<int:aid>/delete', methods=['POST'])
def history_delete(aid):
    if not session.get('user_id'):
        return redirect(url_for('login'))
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('DELETE FROM assessments WHERE id = ? AND user_id = ?', (aid, session['user_id']))
        conn.commit()
    flash("Assessment deleted.", "info")
    return redirect(url_for('history'))

# ===========================================================================
# Authentication Routes (Login & Signup)
# ===========================================================================

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash("Username and password are required.", "warning")
            return render_template('signup.html')

        try:
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute('SELECT id FROM users WHERE username = ?', (username,))
                if c.fetchone():
                    flash("Username already registered. Please choose a different username or log in.", "warning")
                    return render_template('signup.html')

                hashed_password = generate_password_hash(password)
                c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                          (username, email, hashed_password))
                conn.commit()

            flash("Account successfully created! You can now log in.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            flash(f"Database error during registration: {str(e)}", "danger")

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            flash("Please enter both username and password.", "warning")
            return render_template('login.html')

        try:
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute('SELECT id, username, password FROM users WHERE username = ?', (username,))
                user_row = c.fetchone()

                if user_row:
                    user_id, uname, stored_pw = user_row
                    # Check hashed password or fallback to plain text for legacy test accounts
                    is_valid = False
                    if stored_pw.startswith('scrypt:') or stored_pw.startswith('pbkdf2:'):
                        is_valid = check_password_hash(stored_pw, password)
                    else:
                        is_valid = (stored_pw == password)

                    if is_valid:
                        session['user'] = uname
                        session['user_id'] = user_id
                        flash(f"Welcome back, {uname}! Enter the patient's name to begin.", "success")
                        return redirect(url_for('assess_start'))

            flash("Invalid credentials. Please verify your username and password.", "danger")
        except Exception as e:
            flash(f"Authentication error: {str(e)}", "danger")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('user_id', None)
    for k in ('symptom', 'ct_token', 'ct_summary', 'flow', 'saved_key', 'saved_id'):
        session.pop(k, None)
    flash("You have been securely logged out.", "info")
    return redirect(url_for('home'))

# ===========================================================================
# Application Runner
# ===========================================================================
if __name__ == '__main__':
    # Initialize DB before running
    init_db()
    # Run development server
    app.run(debug=True, port=5000)
