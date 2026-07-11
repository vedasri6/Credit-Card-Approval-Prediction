import pandas as pd
import numpy as np

class FeatureEngineer:
    @staticmethod
    def engineer_features(df):
        """Generates domain-specific credit features for model enhancement."""
        df_copy = df.copy()
        
        # 1. Debt-to-Income Ratio (DTI Proxy)
        # Using monthly or annual income. Let's assume Income is monthly and Annual_Income is yearly.
        # Monthly Loan installment estimate: Loan_Amount / 60 (assuming 5-year repayment)
        est_monthly_payment = df_copy['Loan_Amount'] / 60.0
        df_copy['Debt_to_Income_Ratio'] = est_monthly_payment / (df_copy['Income'] + 1.0)
        
        # 2. Employment to Age Ratio
        df_copy['Employment_to_Age_Ratio'] = df_copy['Employment_Years'] / (df_copy['Age'] + 1.0)
        
        # 3. Loan-to-Annual-Income Ratio
        df_copy['Loan_to_Income_Ratio'] = df_copy['Loan_Amount'] / (df_copy['Annual_Income'] + 1.0)
        
        # 4. Monthly vs Annual Income check ratio (should be ~12, but flag discrepancies)
        df_copy['Income_Consistency_Ratio'] = df_copy['Annual_Income'] / (df_copy['Income'] * 12.0 + 1.0)
        
        # 5. Risk score multiplier based on loan purpose and existing loans
        # High loans + existing loans = high risk
        df_copy['Leverage_Score'] = (df_copy['Existing_Loans'] + 1.0) * (df_copy['Loan_Amount'] / (df_copy['Annual_Income'] + 1.0))
        
        # 6. Dependents impact on income
        df_copy['Income_Per_Dependent'] = df_copy['Annual_Income'] / (df_copy['Dependents'] + 1.0)
        
        # Update config columns list dynamically to include these engineered numeric features
        # If we need them scaled, they must be added to NUMERIC_FEATURES.
        # Let's do that or scale them in preprocessing automatically.
        return df_copy

    @staticmethod
    def get_engineered_feature_names():
        """Returns the list of engineered feature names."""
        return [
            'Debt_to_Income_Ratio',
            'Employment_to_Age_Ratio',
            'Loan_to_Income_Ratio',
            'Income_Consistency_Ratio',
            'Leverage_Score',
            'Income_Per_Dependent'
        ]
