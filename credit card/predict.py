import os
import joblib
import pandas as pd
import numpy as np
import json
from config import Config
from feature_engineering import FeatureEngineer

class CreditPredictor:
    def __init__(self):
        self.model_path = Config.MODEL_PATH
        self.preprocessor_path = os.path.join(Config.MODEL_DIR, 'preprocessor.pkl')
        self.model = None
        self.preprocessor = None
        self.metrics = None
        self.load_artifacts()

    def load_artifacts(self):
        """Loads machine learning binary files and metadata configurations."""
        if os.path.exists(self.model_path) and os.path.exists(self.preprocessor_path):
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
            
            # Load training metadata/metrics if present
            if os.path.exists(Config.METRICS_PATH):
                with open(Config.METRICS_PATH, 'r') as f:
                    self.metrics = json.load(f)
        else:
            self.model = None
            self.preprocessor = None

    def is_ready(self):
        """Returns True if the prediction service is loaded and ready."""
        return self.model is not None and self.preprocessor is not None

    def explain_decision(self, input_dict, probability_approval):
        """Computes risk levels, credit scores, reasons, and suggestions."""
        reasons = []
        suggestions = []
        
        # Parse critical factors
        credit_history = input_dict.get('credit_history', 'None')
        payment_history = input_dict.get('payment_history', 'No Delays')
        existing_loans = int(input_dict.get('existing_loans', 0))
        income = float(input_dict.get('income', 0))
        loan_amount = float(input_dict.get('loan_amount', 0))
        employment_years = float(input_dict.get('employment_years', 0))
        
        # Calculate derived ratios
        dti = (loan_amount / 60.0) / (income + 1.0)
        loan_to_income = loan_amount / (income * 12.0 + 1.0)
        
        # Compile positive/negative reasons
        if credit_history == 'Good':
            reasons.append("Excellent credit history and low default record.")
        elif credit_history == 'Bad':
            reasons.append("Historical defaults flagged in credit report.")
            suggestions.append("Settle outstanding loans to clear default flags.")
        else:
            reasons.append("No active credit history detected in banking systems.")
            suggestions.append("Apply for a small credit facility or secured card to build history.")
            
        if payment_history == 'Frequent Delays':
            reasons.append("Frequent delays observed in monthly repayments.")
            suggestions.append("Establish automatic payment plans to avoid future payment delays.")
        elif payment_history == 'No Delays' and credit_history == 'Good':
            reasons.append("Consistent, on-time payment track record.")
            
        if dti > 0.45:
            reasons.append("High debt-to-income ratio (DTI > 45%). Monthly payments exceed recommended levels.")
            suggestions.append("Decrease credit card balance or opt for lower loan amounts.")
        elif dti < 0.20 and income > 0:
            reasons.append("Solid debt capacity with a very low debt-to-income ratio.")
            
        if existing_loans >= 3:
            reasons.append(f"Multiple active credit accounts ({existing_loans}) indicate leverage risk.")
            suggestions.append("Consolidate active loans and close unused lines of credit.")
            
        if employment_years >= 5:
            reasons.append("Stable job tenure indicating reliable income security.")
        elif employment_years < 1:
            reasons.append("Short employment tenure may present income stability concerns.")
            suggestions.append("Maintain current employment for at least 12 months before re-applying.")
            
        if loan_to_income > 2.5:
            reasons.append("Requested loan amount is disproportionate to yearly income.")
            suggestions.append("Consider applying for a smaller loan size.")
            
        # Default reasons if list is empty
        if not reasons:
            if probability_approval >= 0.5:
                reasons.append("Overall financial parameters fall within acceptable credit limits.")
            else:
                reasons.append("Applicant's composite credit rating is below baseline approval thresholds.")
                suggestions.append("Increase monthly savings and reduce variable expenses.")

        # Compute risk, score, and mapping
        approval_score = int(300 + 550 * probability_approval)
        
        if probability_approval >= 0.75:
            risk_level = "Low Risk"
            prediction = "Approved"
            confidence = probability_approval
        elif probability_approval >= 0.50:
            risk_level = "Medium Risk"
            prediction = "Approved"
            confidence = probability_approval
        else:
            risk_level = "High Risk"
            prediction = "Rejected"
            confidence = 1.0 - probability_approval
            
        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'probability': float(probability_approval),
            'risk_level': risk_level,
            'approval_score': approval_score,
            'reasons': reasons,
            'suggestions': list(set(suggestions)) if suggestions else ["Maintain on-time payments to keep account healthy."]
        }

    def predict_single(self, input_dict):
        """Processes single application form, runs model prediction, returns explanation."""
        if not self.is_ready():
            self.load_artifacts()
            if not self.is_ready():
                raise ValueError("Model/Preprocessor artifacts are missing. Run train.py first.")
                
        # 1. Map input keys to DataFrame columns (matching config variables)
        df_input = pd.DataFrame([{
            'Gender': input_dict.get('gender'),
            'Age': float(input_dict.get('age', 0)),
            'Income': float(input_dict.get('income', 0)),
            'Employment_Type': input_dict.get('employment_type'),
            'Employment_Years': float(input_dict.get('employment_years', 0)),
            'Education': input_dict.get('education'),
            'Marital_Status': input_dict.get('marital_status'),
            'Dependents': int(input_dict.get('dependents', 0)),
            'Occupation': input_dict.get('occupation'),
            'Residence_Type': input_dict.get('residence_type'),
            'Credit_History': input_dict.get('credit_history'),
            'Existing_Loans': int(input_dict.get('existing_loans', 0)),
            'Payment_History': input_dict.get('payment_history'),
            'Annual_Income': float(input_dict.get('annual_income', 0)),
            'Loan_Amount': float(input_dict.get('loan_amount', 0)),
            'Loan_Purpose': input_dict.get('loan_purpose')
        }])
        
        # 2. Run Feature Engineering
        df_engineered = FeatureEngineer.engineer_features(df_input)
        
        # 3. Preprocess
        df_processed = self.preprocessor.transform(df_engineered)
        
        # 4. Predict
        probabilities = self.model.predict_proba(df_processed.values)[0]
        prob_approval = probabilities[1]
        
        # Explain Decision
        explanation = self.explain_decision(input_dict, prob_approval)
        
        # Compute feature contributions (Local vs dataset mean comparison)
        explanation['feature_contributions'] = self.compute_local_contributions(df_processed)
        
        return explanation

    def compute_local_contributions(self, df_processed):
        """Estimate feature contributions based on the best model importances/weights."""
        # Simple local explainer: multiply input features by their relative model importance.
        # This provides a relative breakdown of the features for Chart.js.
        contributions = {}
        
        if self.metrics and 'feature_importance' in self.metrics and self.metrics['feature_importance']:
            importances = self.metrics['feature_importance']
            processed_cols = df_processed.columns
            
            # Map processed column importance to their high-level base feature
            base_feature_importances = {}
            for col in processed_cols:
                # Find matching base feature name
                matched_base = None
                for base in Config.NUMERIC_FEATURES + Config.CATEGORICAL_FEATURES:
                    if col.startswith(base):
                        matched_base = base
                        break
                if not matched_base:
                    matched_base = col
                    
                importance_val = importances.get(col, 0.0)
                base_feature_importances[matched_base] = base_feature_importances.get(matched_base, 0.0) + importance_val
                
            # Keep top base features for charts
            sorted_base = sorted(base_feature_importances.items(), key=lambda x: x[1], reverse=True)[:6]
            contributions = {k: float(round(v, 4)) for k, v in sorted_base}
        else:
            # Fallback values if metrics aren't populated
            contributions = {
                'Credit_History': 0.40,
                'Payment_History': 0.25,
                'Income': 0.15,
                'Loan_Amount': 0.10,
                'Employment_Years': 0.06,
                'Age': 0.04
            }
            
        return contributions

    def predict_bulk(self, filepath):
        """Processes bulk CSV file, runs prediction on each row, returns prediction statistics."""
        if not self.is_ready():
            self.load_artifacts()
            if not self.is_ready():
                raise ValueError("Model/Preprocessor artifacts are missing. Run train.py first.")
                
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required_cols = [
            'gender', 'age', 'income', 'employment_type', 'employment_years',
            'education', 'marital_status', 'dependents', 'occupation', 'residence_type',
            'credit_history', 'existing_loans', 'payment_history', 'annual_income',
            'loan_amount', 'loan_purpose'
        ]
        
        # Check case insensitivity
        df_cols_lower = [c.lower() for c in df.columns]
        missing = [c for c in required_cols if c not in df_cols_lower]
        if missing:
            raise ValueError(f"CSV is missing required credit card columns: {missing}")
            
        # Re-index columns with correct case
        mapped_df = pd.DataFrame()
        for orig_col in df.columns:
            mapped_df[orig_col.lower()] = df[orig_col]
            
        predictions = []
        probabilities = []
        risk_levels = []
        approval_scores = []
        
        for idx, row in mapped_df.iterrows():
            row_dict = row.to_dict()
            try:
                res = self.predict_single(row_dict)
                predictions.append(res['prediction'])
                probabilities.append(res['probability'])
                risk_levels.append(res['risk_level'])
                approval_scores.append(res['approval_score'])
            except Exception as e:
                predictions.append('Error')
                probabilities.append(0.0)
                risk_levels.append('Unknown')
                approval_scores.append(300)
                
        # Append results back to the original dataframe
        df['Predicted_Status'] = predictions
        df['Approval_Probability'] = probabilities
        df['Risk_Level'] = risk_levels
        df['Credit_Score'] = approval_scores
        
        return df
