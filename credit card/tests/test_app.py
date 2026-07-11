import os
import pytest
import tempfile
import json
from app import app
from database import DatabaseManager
from config import Config

@pytest.fixture
def client():
    """Configures a Flask test client with an in-memory or temp SQLite DB."""
    # Create a temporary database file
    db_fd, temp_db_path = tempfile.mkstemp()
    
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF in tests for convenience
    
    # Override configuration parameters
    old_db_path = Config.DB_PATH
    Config.DB_PATH = temp_db_path
    
    # Initialize DB schema
    DatabaseManager.init_db()
    
    with app.test_client() as client:
        yield client
        
    # Teardown: close and remove the temporary DB file
    os.close(db_fd)
    os.unlink(temp_db_path)
    Config.DB_PATH = old_db_path

def test_home_page(client):
    """Verifies that the index home page loads correctly."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"AEGIS CREDIT" in response.data

def test_predict_page_get(client):
    """Verifies that the applicant evaluation form page renders."""
    response = client.get('/predict')
    assert response.status_code == 200
    assert b"Credit Card Application Evaluator" in response.data

def test_dashboard_page(client):
    """Verifies that the analytics dashboard page renders."""
    response = client.get('/dashboard')
    assert response.status_code == 200
    assert b"Analytics & Log History" in response.data

def test_about_page(client):
    """Verifies that the model benchmarks details page renders."""
    response = client.get('/about')
    assert response.status_code == 200
    assert b"About Aegis Decision Platform" in response.data

def test_api_docs_page(client):
    """Verifies that the REST API developer documentation renders."""
    response = client.get('/api/docs')
    assert response.status_code == 200
    assert b"REST API Integration" in response.data

def test_404_handling(client):
    """Verifies that invalid URL paths return custom 404 page."""
    response = client.get('/this-path-does-not-exist')
    assert response.status_code == 404
    assert b"404 Error" in response.data
    assert b"Decision Path Not Found" in response.data

def test_database_logging(client):
    """Verifies that database insertion and retrieval mechanisms work."""
    mock_log = {
        'gender': 'Female',
        'age': 28.0,
        'income': 4000.0,
        'employment_type': 'Salaried',
        'employment_years': 3.0,
        'education': 'Graduate',
        'marital_status': 'Single',
        'dependents': 0,
        'occupation': 'Clerk',
        'residence_type': 'Rented',
        'credit_history': 'Good',
        'existing_loans': 0,
        'payment_history': 'No Delays',
        'annual_income': 48000.0,
        'loan_amount': 10000.0,
        'loan_purpose': 'Personal',
        'prediction': 'Approved',
        'confidence': 0.88,
        'probability': 0.88,
        'risk_level': 'Low Risk',
        'approval_score': 784.0
    }
    
    # Insert to database
    last_id = DatabaseManager.insert_prediction(mock_log)
    assert last_id is not None
    assert last_id > 0
    
    # Retrieve
    rows, count = DatabaseManager.get_predictions(limit=10)
    assert count == 1
    assert rows[0]['gender'] == 'Female'
    assert rows[0]['prediction'] == 'Approved'
    assert rows[0]['approval_score'] == 784.0
