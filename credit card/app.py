import os
import logging
import json
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, flash, session
from flask_wtf.csrf import CSRFProtect, generate_csrf
from config import Config
from database import DatabaseManager
from predict import CreditPredictor

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Enable CSRF Protection globally
csrf = CSRFProtect(app)

# Setup Logging
logging.basicConfig(
    filename=Config.LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize database schema
DatabaseManager.init_db()

# Load Prediction Engine
predictor = CreditPredictor()
if not predictor.is_ready():
    logging.warning("Model and Preprocessor binaries not found. Training required before serving predictions.")

# Helper to check allowed file extensions for bulk upload
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# Custom helper function to check if static visual files exist in templates
@app.context_processor
def utility_processor():
    def os_exists(path):
        return os.path.exists(os.path.join(app.static_folder, path.replace('static/', '')))
    return dict(os_exists=os_exists)

# ----------------- ROUTES -----------------

@app.route('/')
def home():
    """Renders the Aegis index home screen."""
    metrics = None
    if os.path.exists(Config.METRICS_PATH):
        try:
            with open(Config.METRICS_PATH, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            logging.error(f"Error reading metrics JSON: {str(e)}")
            
    return render_template('index.html', metrics=metrics)

@app.route('/predict', methods=['GET'])
def predict():
    """Renders single application form."""
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict_post():
    """Processes single form submission, runs ML model, records result in DB."""
    if not predictor.is_ready():
        flash("Machine Learning models are not trained yet. Run train.py first.", "danger")
        return redirect(url_for('predict'))
        
    try:
        # 1. Fetch form inputs with XSS prevention/typecasting
        form_data = {
            'gender': request.form.get('gender'),
            'age': float(request.form.get('age', 0)),
            'income': float(request.form.get('income', 0)),
            'employment_type': request.form.get('employment_type'),
            'employment_years': float(request.form.get('employment_years', 0)),
            'education': request.form.get('education'),
            'marital_status': request.form.get('marital_status'),
            'dependents': int(request.form.get('dependents', 0)),
            'occupation': request.form.get('occupation'),
            'residence_type': request.form.get('residence_type'),
            'credit_history': request.form.get('credit_history'),
            'existing_loans': int(request.form.get('existing_loans', 0)),
            'payment_history': request.form.get('payment_history'),
            'annual_income': float(request.form.get('annual_income', 0)),
            'loan_amount': float(request.form.get('loan_amount', 0)),
            'loan_purpose': request.form.get('loan_purpose')
        }
        
        # 2. Run prediction model pipeline
        result = predictor.predict_single(form_data)
        
        # Merge input and prediction fields to log
        log_data = {**form_data, **result}
        
        # 3. Store record in DB history
        DatabaseManager.insert_prediction(log_data)
        logging.info(f"Successful prediction executed. Decision: {result['prediction']} | Score: {result['approval_score']}")
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        logging.error(f"Error during single prediction processing: {str(e)}")
        flash(f"Application processing failed: {str(e)}", "danger")
        return redirect(url_for('predict'))

@app.route('/dashboard')
def dashboard():
    """Renders analytical logs and historical statistics page."""
    search_query = request.args.get('search', '')
    page = int(request.args.get('page', 1))
    limit = 10
    offset = (page - 1) * limit
    
    try:
        # Retrieve paginated rows
        rows, total_count = DatabaseManager.get_predictions(limit, offset, search_query)
        total_pages = (total_count + limit - 1) // limit
        
        # Retrieve aggregate metrics
        analytics = DatabaseManager.get_analytics()
        
        return render_template(
            'dashboard.html',
            rows=rows,
            analytics=analytics,
            current_page=page,
            total_pages=total_pages,
            search_query=search_query
        )
    except Exception as e:
        logging.error(f"Error loading dashboard: {str(e)}")
        flash("Failed to load dashboard statistics.", "danger")
        return render_template('404.html')

@app.route('/predict/bulk', methods=['POST'])
def predict_bulk_post():
    """Ingests bulk credit application spreadsheets, runs ML, and serves downloadable outcomes."""
    if not predictor.is_ready():
        flash("Machine Learning models are not trained yet.", "danger")
        return redirect(url_for('dashboard'))
        
    if 'file' not in request.files:
        flash("No spreadsheet file uploaded.", "danger")
        return redirect(url_for('dashboard'))
        
    file = request.files['file']
    if file.filename == '':
        flash("No file selected.", "danger")
        return redirect(url_for('dashboard'))
        
    if file and allowed_file(file.filename):
        try:
            # Save uploaded CSV locally
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
            upload_path = os.path.join(Config.UPLOAD_FOLDER, 'bulk_upload.csv')
            file.save(upload_path)
            
            # Execute bulk calculations
            annotated_df = predictor.predict_bulk(upload_path)
            
            # Save annotated results
            results_path = os.path.join(Config.UPLOAD_FOLDER, 'bulk_results.csv')
            annotated_df.to_csv(results_path, index=False)
            
            # Log bulk evaluations to database history directory
            for idx, row in annotated_df.iterrows():
                try:
                    # Parse into DB schema format
                    db_row = {
                        'gender': row.get('gender'),
                        'age': int(row.get('age', 0)),
                        'income': float(row.get('income', 0)),
                        'employment_type': row.get('employment_type'),
                        'employment_years': float(row.get('employment_years', 0)),
                        'education': row.get('education'),
                        'marital_status': row.get('marital_status'),
                        'dependents': int(row.get('dependents', 0)),
                        'occupation': row.get('occupation'),
                        'residence_type': row.get('residence_type'),
                        'credit_history': row.get('credit_history'),
                        'existing_loans': int(row.get('existing_loans', 0)),
                        'payment_history': row.get('payment_history'),
                        'annual_income': float(row.get('annual_income', 0)),
                        'loan_amount': float(row.get('loan_amount', 0)),
                        'loan_purpose': row.get('loan_purpose'),
                        'prediction': row.get('Predicted_Status'),
                        'confidence': float(row.get('Approval_Probability', 0.5)),
                        'probability': float(row.get('Approval_Probability', 0.5)),
                        'risk_level': row.get('Risk_Level'),
                        'approval_score': float(row.get('Credit_Score', 300))
                    }
                    DatabaseManager.insert_prediction(db_row)
                except Exception as e:
                    logging.error(f"Error logging bulk row {idx} to DB: {str(e)}")
                    
            logging.info(f"Bulk prediction complete. Processed {len(annotated_df)} applications.")
            flash(f"Successfully processed {len(annotated_df)} applications.", "success")
            
            # Serve file download attachment
            return send_file(
                results_path,
                mimetype='text/csv',
                as_attachment=True,
                download_name='annotated_credit_predictions.csv'
            )
            
        except Exception as e:
            logging.error(f"Bulk ingestion error: {str(e)}")
            flash(f"Bulk evaluation failed: {str(e)}", "danger")
            return redirect(url_for('dashboard'))
    else:
        flash("Invalid file extension. Only .csv spreadsheets are accepted.", "danger")
        return redirect(url_for('dashboard'))

@app.route('/download-template')
def download_csv_template():
    """Generates sample credit application CSV layout structure for user guidelines."""
    cols = [
        'gender', 'age', 'income', 'employment_type', 'employment_years',
        'education', 'marital_status', 'dependents', 'occupation', 'residence_type',
        'credit_history', 'existing_loans', 'payment_history', 'annual_income',
        'loan_amount', 'loan_purpose'
    ]
    sample_data = [{
        'gender': 'Male',
        'age': 32,
        'income': 4800,
        'employment_type': 'Salaried',
        'employment_years': 4.5,
        'education': 'Graduate',
        'marital_status': 'Married',
        'dependents': 1,
        'occupation': 'Engineer',
        'residence_type': 'Rented',
        'credit_history': 'Good',
        'existing_loans': 1,
        'payment_history': 'No Delays',
        'annual_income': 58000,
        'loan_amount': 25000,
        'loan_purpose': 'Home'
    }]
    
    template_path = os.path.join(Config.UPLOAD_FOLDER, 'credit_bulk_template.csv')
    pd.DataFrame(sample_data, columns=cols).to_csv(template_path, index=False)
    
    return send_file(
        template_path,
        mimetype='text/csv',
        as_attachment=True,
        download_name='credit_bulk_template.csv'
    )

@app.route('/download-history')
def download_history_csv():
    """Exports prediction history database logs as downloadable CSV spreadsheet."""
    try:
        conn = DatabaseManager.get_connection()
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
        conn.close()
        
        export_path = os.path.join(Config.UPLOAD_FOLDER, 'credit_history_export.csv')
        df.to_csv(export_path, index=False)
        
        return send_file(
            export_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name='credit_approval_history_logs.csv'
        )
    except Exception as e:
        logging.error(f"Error exporting history log: {str(e)}")
        flash("Exporting history database failed.", "danger")
        return redirect(url_for('dashboard'))

@app.route('/about')
def about():
    """Renders models evaluation comparative leaderboard."""
    metrics = None
    if os.path.exists(Config.METRICS_PATH):
        try:
            with open(Config.METRICS_PATH, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            logging.error(f"Error reading metrics summary: {str(e)}")
            
    return render_template('about.html', metrics=metrics)

# ----------------- REST API ENDPOINTS & DOCS -----------------

@app.route('/api/predict', methods=['POST'])
@csrf.exempt  # Exempt from web UI CSRF token to allow external API clients
def api_predict():
    """REST API endpoint for automated inference. Secure, validated, and JSON safe."""
    if not predictor.is_ready():
        return jsonify({'error': 'Prediction model is not loaded.'}), 503
        
    try:
        req_json = request.json
        if not req_json:
            return jsonify({'error': 'JSON payload body required.'}), 400
            
        # Basic validation checklist
        req_fields = [
            'gender', 'age', 'income', 'employment_type', 'employment_years',
            'education', 'marital_status', 'dependents', 'occupation', 'residence_type',
            'credit_history', 'existing_loans', 'payment_history', 'annual_income',
            'loan_amount', 'loan_purpose'
        ]
        missing = [f for f in req_fields if f not in req_json]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400
            
        result = predictor.predict_single(req_json)
        
        # Log payload to database
        db_log = {**req_json, **result}
        DatabaseManager.insert_prediction(db_log)
        
        return jsonify({
            'status': 'success',
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'probability': result['probability'],
            'risk_level': result['risk_level'],
            'approval_score': result['approval_score'],
            'reasons': result['reasons']
        })
    except Exception as e:
        logging.error(f"API Error: {str(e)}")
        return jsonify({'error': f'Prediction execution failed: {str(e)}'}), 500

@app.route('/api/docs')
def api_docs():
    """Renders custom Swagger-style testing panel inside the application layout."""
    return render_template('api_docs.html')

@app.route('/api/csrf-token', methods=['GET'])
def get_csrf():
    """Endpoint to fetch CSRF token (useful for SPA integrations)."""
    return jsonify({'csrf_token': generate_csrf()})

# ----------------- ERROR HANDLING -----------------

@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 Route Handler."""
    return render_template('404.html'), 404

if __name__ == '__main__':
    # Run the web application
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
