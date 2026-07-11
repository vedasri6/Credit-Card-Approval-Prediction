import os

class Config:
    # Base directory of the application
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # Secret key for sessions
    SECRET_KEY = os.environ.get('SECRET_KEY', 'super-secret-key-for-credit-card-approval-ml')
    
    # Database Configuration
    DB_PATH = os.path.join(BASE_DIR, 'database.db')
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_PATH}"
    
    # Dataset and Model Configuration
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    DATASET_PATH = os.path.join(DATASET_DIR, 'credit_card_data.csv')
    
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
    SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
    ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder.pkl')
    METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')
    
    # Static visual assets
    VISUALS_DIR = os.path.join(BASE_DIR, 'static', 'images')
    
    # Logging Configuration
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    LOG_PATH = os.path.join(LOG_DIR, 'app.log')
    
    # CSV Upload configuration
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    ALLOWED_EXTENSIONS = {'csv'}
    
    # Model Training Parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Feature columns used for processing and training
    NUMERIC_FEATURES = [
        'Age', 'Income', 'Employment_Years', 'Dependents', 
        'Existing_Loans', 'Loan_Amount', 'Annual_Income',
        'Debt_to_Income_Ratio', 'Employment_to_Age_Ratio', 
        'Loan_to_Income_Ratio', 'Income_Consistency_Ratio', 
        'Leverage_Score', 'Income_Per_Dependent'
    ]
    
    CATEGORICAL_FEATURES = [
        'Gender', 'Employment_Type', 'Education', 'Marital_Status', 
        'Occupation', 'Residence_Type', 'Credit_History', 
        'Payment_History', 'Loan_Purpose'
    ]
    
    TARGET_COLUMN = 'Approved'

# Ensure directories exist
for path in [Config.DATASET_DIR, Config.MODEL_DIR, Config.VISUALS_DIR, Config.LOG_DIR, Config.UPLOAD_FOLDER]:
    os.makedirs(path, exist_ok=True)
