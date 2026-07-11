import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix, 
    classification_report
)
from sklearn.model_selection import cross_val_score, learning_curve
from config import Config

class ModelEvaluator:
    @staticmethod
    def evaluate_model(model, X_train, y_train, X_test, y_test, cv_folds=5):
        """Computes exhaustive metrics for a trained classifier."""
        # Predictions
        y_pred = model.predict(X_test)
        
        # Probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            dec_func = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-dec_func)) # Sigmoid mapping
        else:
            y_prob = y_pred.astype(float)
            
        # Standard metrics
        metrics = {
            'Accuracy': float(accuracy_score(y_test, y_pred)),
            'Precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'Recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'F1_Score': float(f1_score(y_test, y_pred, zero_division=0)),
            'ROC_AUC': float(roc_auc_score(y_test, y_prob)),
        }
        
        # Cross validation on training data to assess generalization
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
        metrics['CV_Accuracy_Mean'] = float(np.mean(cv_scores))
        metrics['CV_Accuracy_Std'] = float(np.std(cv_scores))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['Confusion_Matrix'] = cm.tolist() # [[TN, FP], [FN, TP]]
        
        # Classification report as dict
        metrics['Classification_Report'] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        return metrics, y_prob

    @staticmethod
    def get_feature_importance(model, feature_names):
        """Extracts and normalizes feature importance scores or coefficients."""
        importance_dict = {}
        
        # Tree-based and XGBoost model feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
        # Linear models (Logistic Regression) coefficients
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
            # Normalize to sum up to 1 for comparison consistency
            if importances.sum() > 0:
                importances = importances / importances.sum()
                
        # SVM or non-linear models that do not directly expose importances
        else:
            # We skip or return empty (or handle in visualization)
            return None
            
        for name, imp in zip(feature_names, importances):
            importance_dict[name] = float(imp)
            
        # Sort by importance descending
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        return sorted_importance

    @staticmethod
    def get_learning_curve_data(model, X, y, cv_folds=5):
        """Computes learning curve data points for visual inspection."""
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv_folds, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5), 
            random_state=Config.RANDOM_STATE
        )
        return {
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
            'test_scores_mean': np.mean(test_scores, axis=1).tolist()
        }
