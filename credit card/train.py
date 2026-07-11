import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import Config
from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model import ModelManager
from evaluation import ModelEvaluator
from visualization import DataVisualizer

def generate_synthetic_dataset(num_samples=1500):
    """Generates a highly realistic credit card application dataset."""
    np.random.seed(Config.RANDOM_STATE)
    
    # 1. Base demographics and background
    genders = np.random.choice(['Male', 'Female'], size=num_samples, p=[0.55, 0.45])
    ages = np.random.randint(18, 70, size=num_samples)
    
    educations = np.random.choice(
        ['High School', 'Graduate', 'Post-Graduate'], 
        size=num_samples, p=[0.3, 0.5, 0.2]
    )
    
    marital_statuses = np.random.choice(
        ['Single', 'Married', 'Divorced', 'Widowed'], 
        size=num_samples, p=[0.35, 0.5, 0.1, 0.05]
    )
    
    dependents = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    # 2. Employment
    employment_types = np.random.choice(
        ['Salaried', 'Self-Employed', 'Retired', 'Unemployed'],
        size=num_samples, p=[0.6, 0.25, 0.1, 0.05]
    )
    
    # Make occupation dependent on education
    occupations = []
    for edu in educations:
        if edu == 'Post-Graduate':
            occupations.append(np.random.choice(['Manager', 'Engineer', 'Business Owner'], p=[0.4, 0.5, 0.1]))
        elif edu == 'Graduate':
            occupations.append(np.random.choice(['Manager', 'Engineer', 'Clerk', 'Sales', 'Business Owner'], p=[0.2, 0.3, 0.2, 0.2, 0.1]))
        else:
            occupations.append(np.random.choice(['Clerk', 'Sales', 'Others'], p=[0.3, 0.4, 0.3]))
    occupations = np.array(occupations)
    
    # Adjust employment years based on age and employment status
    employment_years = []
    for i in range(num_samples):
        age = ages[i]
        emp_type = employment_types[i]
        if emp_type == 'Unemployed':
            employment_years.append(0.0)
        elif emp_type == 'Retired':
            max_workable = max(1, age - 18)
            min_workable = min(15, max_workable)
            if min_workable >= max_workable:
                work_years = max_workable
            else:
                work_years = np.random.randint(min_workable, max_workable + 1)
            employment_years.append(float(work_years))
        else:
            max_workable = max(1, min(30, age - 18))
            work_years = np.random.randint(0, max_workable + 1)
            employment_years.append(float(work_years))
    employment_years = np.array(employment_years)
    
    # 3. Finances
    residence_types = np.random.choice(
        ['Owned', 'Rented', 'Mortgaged'], 
        size=num_samples, p=[0.3, 0.5, 0.2]
    )
    
    # Income based on age, education, and occupation
    incomes = []
    for i in range(num_samples):
        age = ages[i]
        emp_type = employment_types[i]
        occ = occupations[i]
        
        if emp_type == 'Unemployed':
            base_income = np.random.randint(500, 1500)
        elif emp_type == 'Retired':
            base_income = np.random.randint(1500, 4000)
        else:
            if occ in ['Manager', 'Business Owner']:
                base_income = np.random.randint(6000, 15000)
            elif occ == 'Engineer':
                base_income = np.random.randint(4500, 10000)
            elif occ in ['Clerk', 'Sales']:
                base_income = np.random.randint(2500, 5500)
            else:
                base_income = np.random.randint(1500, 3500)
                
        # Add age progression factor
        age_bonus = (age - 18) * 40
        incomes.append(float(base_income + age_bonus))
    incomes = np.array(incomes)
    
    annual_incomes = incomes * 12.0 + np.random.randint(-5000, 5000, size=num_samples)
    annual_incomes = np.clip(annual_incomes, 10000, None)
    
    # Loans
    existing_loans = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.5, 0.35, 0.1, 0.05])
    
    loan_purposes = np.random.choice(
        ['Home', 'Education', 'Personal', 'Business', 'Car'], 
        size=num_samples, p=[0.25, 0.2, 0.25, 0.2, 0.1]
    )
    
    # Loan amount requested relates to income
    loan_amounts = []
    for i in range(num_samples):
        ann_inc = annual_incomes[i]
        purp = loan_purposes[i]
        if purp == 'Home':
            mult = np.random.uniform(1.5, 3.5)
        elif purp == 'Business':
            mult = np.random.uniform(0.5, 1.8)
        else:
            mult = np.random.uniform(0.1, 0.6)
        loan_amounts.append(float(round(ann_inc * mult / 1000) * 1000))
    loan_amounts = np.array(loan_amounts)
    
    # 4. Credit History and Payment History (Key predictive factors)
    credit_histories = np.random.choice(
        ['Good', 'Bad', 'None'], 
        size=num_samples, p=[0.7, 0.2, 0.1]
    )
    
    payment_histories = []
    for i in range(num_samples):
        ch = credit_histories[i]
        if ch == 'Good':
            payment_histories.append(np.random.choice(['No Delays', 'Occasional Delays'], p=[0.8, 0.2]))
        elif ch == 'Bad':
            payment_histories.append(np.random.choice(['Occasional Delays', 'Frequent Delays'], p=[0.3, 0.7]))
        else:
            payment_histories.append('No Delays')
    payment_histories = np.array(payment_histories)
    
    # 5. Label scoring (True credit risk formula)
    approved_labels = []
    for i in range(num_samples):
        score = 0.0
        
        # Credit history is massive
        if credit_histories[i] == 'Good':
            score += 3.5
        elif credit_histories[i] == 'Bad':
            score -= 4.0
        else:
            score += 0.5
            
        # Payment delays
        if payment_histories[i] == 'No Delays':
            score += 1.5
        elif payment_histories[i] == 'Frequent Delays':
            score -= 3.0
            
        # Debt-to-income
        dti = (loan_amounts[i] / 60.0) / incomes[i]
        if dti > 0.45:
            score -= 2.5
        elif dti < 0.20:
            score += 1.0
            
        # Income and employment duration
        if annual_incomes[i] > 80000:
            score += 1.0
        elif annual_incomes[i] < 20000:
            score -= 1.5
            
        if employment_years[i] > 5:
            score += 1.2
        elif employment_years[i] < 1:
            score -= 1.0
            
        # Existing obligations
        if existing_loans[i] >= 2:
            score -= 1.2
            
        # Education and residence
        if educations[i] in ['Graduate', 'Post-Graduate']:
            score += 0.5
        if residence_types[i] == 'Owned':
            score += 0.7
            
        # Add gaussian noise
        noise = np.random.normal(0, 1.0)
        score += noise
        
        # Threshold for approval
        if score >= 1.2:
            approved_labels.append(1)
        else:
            approved_labels.append(0)
            
    # Combine to DataFrame
    dataset = pd.DataFrame({
        'Gender': genders,
        'Age': ages,
        'Income': incomes,
        'Employment_Type': employment_types,
        'Employment_Years': employment_years,
        'Education': educations,
        'Marital_Status': marital_statuses,
        'Dependents': dependents,
        'Occupation': occupations,
        'Residence_Type': residence_types,
        'Credit_History': credit_histories,
        'Existing_Loans': existing_loans,
        'Payment_History': payment_histories,
        'Annual_Income': annual_incomes,
        'Loan_Amount': loan_amounts,
        'Loan_Purpose': loan_purposes,
        'Approved': approved_labels
    })
    
    # Save to file
    dataset.to_csv(Config.DATASET_PATH, index=False)
    print(f"Generated synthetic dataset with {len(dataset)} records at: {Config.DATASET_PATH}")
    return dataset

def run_training_pipeline():
    print("=" * 60)
    print("STARTING CREDIT CARD APPROVAL ML PIPELINE TRAINING")
    print("=" * 60)
    
    # 1. Load or Generate Dataset
    if not os.path.exists(Config.DATASET_PATH):
        df = generate_synthetic_dataset()
    else:
        df = pd.read_csv(Config.DATASET_PATH)
        print(f"Loaded existing dataset from {Config.DATASET_PATH}")
        
    # Output Dataset Summary
    print("\n[DATASET SUMMARY]")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing Values:\n{df.isnull().sum().to_dict()}")
    print(f"Duplicate Values: {df.duplicated().sum()}")
    print(f"Target Distribution:\n{df['Approved'].value_counts(normalize=True).to_dict()}")
    
    # Save exploratory graphs
    print("\nGenerating exploratory graphs...")
    DataVisualizer.save_correlation_matrix(df)
    DataVisualizer.save_distribution_plots(df)
    DataVisualizer.save_count_and_pie_plots(df)
    DataVisualizer.save_scatter_and_box_plots(df)
    DataVisualizer.save_pairplot(df)
    
    # 2. Feature Engineering
    print("\nRunning Feature Engineering...")
    df_engineered = FeatureEngineer.engineer_features(df)
    
    # 3. Preprocessing
    print("Initializing Data Preprocessor and Split...")
    X = df_engineered.drop(columns=[Config.TARGET_COLUMN])
    y = df_engineered[Config.TARGET_COLUMN]
    
    # Train-test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
    )
    
    preprocessor = DataPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)
    
    feature_names = X_train_processed.columns.tolist()
    
    # 4. Model Training & Evaluation
    print("\nTraining and evaluating classifiers...")
    models = ModelManager.get_all_models()
    
    results = {}
    models_probs = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f" -> Training {name}...")
        model = ModelManager.train_model(model, X_train_processed.values, y_train.values)
        trained_models[name] = model
        
        # Evaluate
        metrics, y_prob = ModelEvaluator.evaluate_model(
            model, X_train_processed.values, y_train.values, 
            X_test_processed.values, y_test.values
        )
        
        results[name] = metrics
        models_probs[name] = y_prob
        print(f"    Validation Accuracy: {metrics['Accuracy']:.4f} | ROC-AUC: {metrics['ROC_AUC']:.4f}")
        
    # Generate combined ROC benchmarking plot
    DataVisualizer.save_roc_curves(models_probs, y_test)
    
    # 5. Leaderboard and Best Model Selection
    leaderboard = []
    for name, metrics in results.items():
        leaderboard.append({
            'Model': name,
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1_Score': metrics['F1_Score'],
            'ROC_AUC': metrics['ROC_AUC'],
            'CV_Accuracy': metrics['CV_Accuracy_Mean']
        })
    leaderboard_df = pd.DataFrame(leaderboard).sort_values(by='Accuracy', ascending=False)
    print("\n[MODEL LEADERBOARD]")
    print(leaderboard_df.to_string(index=False))
    
    best_model_name = leaderboard_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]
    print(f"\n*** SELECTED BEST MODEL: {best_model_name} (Acc: {best_metrics['Accuracy']:.4f}) ***")
    
    # Save Best Model specific visualizations
    importance = ModelEvaluator.get_feature_importance(best_model, feature_names)
    if importance:
        DataVisualizer.save_feature_importance_plot(importance)
        
    cm = np.array(best_metrics['Confusion_Matrix'])
    DataVisualizer.save_confusion_matrix_plot(cm)
    
    # Save objects to artifacts
    joblib.dump(best_model, Config.MODEL_PATH)
    joblib.dump(preprocessor, os.path.join(Config.MODEL_DIR, 'preprocessor.pkl')) # Save preprocessor class directly
    
    # Save leaderboard and performance metrics JSON
    metrics_summary = {
        'best_model': best_model_name,
        'leaderboard': leaderboard,
        'best_model_metrics': best_metrics,
        'feature_importance': importance
    }
    with open(Config.METRICS_PATH, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
        
    print(f"\nSuccessfully serialized artifacts:")
    print(f" - Model: {Config.MODEL_PATH}")
    print(f" - Preprocessor: {os.path.join(Config.MODEL_DIR, 'preprocessor.pkl')}")
    print(f" - Metrics summary: {Config.METRICS_PATH}")
    print("\nTraining Pipeline successfully complete!\n")

if __name__ == '__main__':
    run_training_pipeline()
