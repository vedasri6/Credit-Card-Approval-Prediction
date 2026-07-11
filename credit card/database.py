import sqlite3
import datetime
import os
from config import Config

class DatabaseManager:
    @staticmethod
    def get_connection():
        """Establishes connection to the SQLite database."""
        conn = sqlite3.connect(Config.DB_PATH)
        conn.row_factory = sqlite3.Row  # Returns rows as dictionary-like objects
        return conn

    @staticmethod
    def init_db():
        """Initializes database tables if they do not exist."""
        conn = DatabaseManager.get_connection()
        cursor = conn.cursor()
        
        # Create predictions history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                gender TEXT,
                age REAL,
                income REAL,
                employment_type TEXT,
                employment_years REAL,
                education TEXT,
                marital_status TEXT,
                dependents INTEGER,
                occupation TEXT,
                residence_type TEXT,
                credit_history TEXT,
                existing_loans INTEGER,
                payment_history TEXT,
                annual_income REAL,
                loan_amount REAL,
                loan_purpose TEXT,
                prediction TEXT,
                confidence REAL,
                probability REAL,
                risk_level TEXT,
                approval_score REAL
            )
        ''')
        conn.commit()
        conn.close()

    @staticmethod
    def insert_prediction(data):
        """Inserts a single prediction record into the database."""
        conn = DatabaseManager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (
                gender, age, income, employment_type, employment_years, 
                education, marital_status, dependents, occupation, residence_type, 
                credit_history, existing_loans, payment_history, annual_income, 
                loan_amount, loan_purpose, prediction, confidence, probability, 
                risk_level, approval_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('gender'),
            float(data.get('age', 0)),
            float(data.get('income', 0)),
            data.get('employment_type'),
            float(data.get('employment_years', 0)),
            data.get('education'),
            data.get('marital_status'),
            int(data.get('dependents', 0)),
            data.get('occupation'),
            data.get('residence_type'),
            data.get('credit_history'),
            int(data.get('existing_loans', 0)),
            data.get('payment_history'),
            float(data.get('annual_income', 0)),
            float(data.get('loan_amount', 0)),
            data.get('loan_purpose'),
            data.get('prediction'),
            float(data.get('confidence', 0.0)),
            float(data.get('probability', 0.0)),
            data.get('risk_level'),
            float(data.get('approval_score', 0.0))
        ))
        
        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        return last_id

    @staticmethod
    def get_predictions(limit=100, offset=0, search_query=None):
        """Retrieves prediction history with optional search and pagination."""
        conn = DatabaseManager.get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM predictions"
        params = []
        
        if search_query:
            query += " WHERE prediction LIKE ? OR risk_level LIKE ? OR loan_purpose LIKE ? OR occupation LIKE ?"
            like_val = f"%{search_query}%"
            params.extend([like_val, like_val, like_val, like_val])
            
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        
        # Get total count for pagination
        count_query = "SELECT COUNT(*) FROM predictions"
        count_params = []
        if search_query:
            count_query += " WHERE prediction LIKE ? OR risk_level LIKE ? OR loan_purpose LIKE ? OR occupation LIKE ?"
            count_params.extend([like_val, like_val, like_val, like_val])
            
        cursor.execute(count_query, count_params)
        total_count = cursor.fetchone()[0]
        
        conn.close()
        return rows, total_count

    @staticmethod
    def get_analytics():
        """Computes summary statistics of predictions for the dashboard."""
        conn = DatabaseManager.get_connection()
        cursor = conn.cursor()
        
        analytics = {}
        
        # Total counts
        cursor.execute("SELECT COUNT(*) FROM predictions")
        analytics['total_predictions'] = cursor.fetchone()[0]
        
        # Approved count
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Approved'")
        analytics['approved_count'] = cursor.fetchone()[0]
        
        # Rejected count
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Rejected'")
        analytics['rejected_count'] = cursor.fetchone()[0]
        
        # Average loan amount
        cursor.execute("SELECT AVG(loan_amount) FROM predictions")
        avg_loan = cursor.fetchone()[0]
        analytics['avg_loan_amount'] = round(avg_loan, 2) if avg_loan else 0.0
        
        # Risk distribution
        cursor.execute("SELECT risk_level, COUNT(*) as count FROM predictions GROUP BY risk_level")
        analytics['risk_distribution'] = {row['risk_level']: row['count'] for row in cursor.fetchall()}
        
        # Purpose distribution
        cursor.execute("SELECT loan_purpose, COUNT(*) as count FROM predictions GROUP BY loan_purpose")
        analytics['purpose_distribution'] = {row['loan_purpose']: row['count'] for row in cursor.fetchall()}
        
        # Monthly trend (last 6 months)
        cursor.execute("""
            SELECT strftime('%Y-%m', timestamp) as month, 
                   SUM(CASE WHEN prediction='Approved' THEN 1 ELSE 0 END) as approved,
                   SUM(CASE WHEN prediction='Rejected' THEN 1 ELSE 0 END) as rejected
            FROM predictions 
            GROUP BY month 
            ORDER BY month DESC 
            LIMIT 6
        """)
        analytics['monthly_trends'] = [dict(row) for row in cursor.fetchall()][::-1]
        
        conn.close()
        return analytics
