import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    ExtraTreesClassifier, 
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from config import Config

class ModelManager:
    @staticmethod
    def get_all_models():
        """Returns dictionary of baseline machine learning classifiers."""
        random_state = Config.RANDOM_STATE
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=random_state, max_depth=8),
            'Random Forest': RandomForestClassifier(random_state=random_state, n_estimators=100, max_depth=10, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=random_state, n_estimators=100, learning_rate=0.1),
            'Extra Trees': ExtraTreesClassifier(random_state=random_state, n_estimators=100, max_depth=10, n_jobs=-1),
            'AdaBoost': AdaBoostClassifier(random_state=random_state, n_estimators=100),
            'XGBoost': XGBClassifier(random_state=random_state, n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss', n_jobs=-1),
            'Support Vector Machine': SVC(random_state=random_state, probability=True, kernel='rbf', C=1.0),
            'Naive Bayes': GaussianNB(),
            'K Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        }
        
        # Add Voting Classifier using top-performing estimators
        # Since we haven't trained them yet, we'll compose it from a diverse ensemble:
        # Logistic Regression, Random Forest, XGBoost, and Support Vector Machine
        estimators = [
            ('lr', models['Logistic Regression']),
            ('rf', models['Random Forest']),
            ('xgb', models['XGBoost']),
            ('svm', models['Support Vector Machine'])
        ]
        models['Voting Classifier'] = VotingClassifier(
            estimators=estimators, 
            voting='soft',
            n_jobs=-1
        )
        
        return models

    @staticmethod
    def train_model(model, X_train, y_train):
        """Fits a model on the provided training set."""
        model.fit(X_train, y_train)
        return model
