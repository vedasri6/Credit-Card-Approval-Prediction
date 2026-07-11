import os
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from config import Config

class DataVisualizer:
    @staticmethod
    def setup_style():
        """Sets up aesthetic plotting standards."""
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.titlesize': 16,
            'figure.dpi': 150
        })

    @staticmethod
    def save_correlation_matrix(df, filepath=None):
        """Generates and saves a correlation heatmap for numeric features."""
        DataVisualizer.setup_style()
        plt.figure(figsize=(10, 8))
        
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        # Drop columns with zero variance to avoid NaNs in correlation
        numeric_df = numeric_df.loc[:, numeric_df.std() > 0]
        
        corr = numeric_df.corr()
        
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                    square=True, linewidths=.5, cbar_kws={"shrink": .8})
        plt.title("Correlation Matrix of Numeric Features", pad=20)
        plt.tight_layout()
        
        save_path = filepath or os.path.join(Config.VISUALS_DIR, 'correlation_matrix.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @staticmethod
    def save_distribution_plots(df, filepath_hist=None, filepath_dist=None):
        """Generates histograms and distribution plots for Age and Income."""
        DataVisualizer.setup_style()
        
        # Age Distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x='Age', kde=True, hue='Approved', multiple='stack', palette='muted')
        plt.title("Distribution of Age by Approval Status")
        plt.xlabel("Age")
        plt.ylabel("Count")
        save_path = filepath_hist or os.path.join(Config.VISUALS_DIR, 'age_distribution.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        # Income Distribution (KDE / Distribution plot)
        plt.figure(figsize=(8, 5))
        sns.kdeplot(data=df, x='Annual_Income', hue='Approved', fill=True, common_norm=False, palette='crest', alpha=0.5)
        plt.title("Annual Income Kernel Density Estimate")
        plt.xlabel("Annual Income ($)")
        plt.ylabel("Density")
        save_path = filepath_dist or os.path.join(Config.VISUALS_DIR, 'income_distribution.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @staticmethod
    def save_count_and_pie_plots(df, filepath_count=None, filepath_pie=None):
        """Generates countplot of approval status and pie chart of loan purposes."""
        DataVisualizer.setup_style()
        
        # Target variable count plot
        plt.figure(figsize=(6, 5))
        # Map 1/0 or Approved/Rejected to label strings for visual clarity
        temp_df = df.copy()
        if temp_df['Approved'].dtype in [np.int64, np.int32, int]:
            temp_df['Approval Status'] = temp_df['Approved'].map({1: 'Approved', 0: 'Rejected'})
        else:
            temp_df['Approval Status'] = temp_df['Approved']
            
        sns.countplot(data=temp_df, x='Approval Status', palette=['#2ecc71', '#e74c3c'])
        plt.title("Distribution of Credit Approval Status")
        plt.xlabel("Status")
        plt.ylabel("Application Count")
        save_path = filepath_count or os.path.join(Config.VISUALS_DIR, 'approval_countplot.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        # Loan purpose pie chart
        plt.figure(figsize=(6, 6))
        purpose_counts = df['Loan_Purpose'].value_counts()
        plt.pie(purpose_counts, labels=purpose_counts.index, autopct='%1.1f%%', 
                startangle=140, colors=sns.color_palette('pastel'))
        plt.title("Credit Applications by Loan Purpose")
        save_path = filepath_pie or os.path.join(Config.VISUALS_DIR, 'loan_purpose_pie.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @staticmethod
    def save_scatter_and_box_plots(df, filepath_scatter=None, filepath_box=None):
        """Generates box plots and scatter plots."""
        DataVisualizer.setup_style()
        
        # Boxplot of Income vs Credit History
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x='Credit_History', y='Income', hue='Approved', palette='Set2')
        plt.title("Monthly Income by Credit History and Approval Status")
        plt.xlabel("Credit History Status")
        plt.ylabel("Monthly Income ($)")
        save_path = filepath_box or os.path.join(Config.VISUALS_DIR, 'income_credit_boxplot.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        # Scatter plot of Annual Income vs Loan Amount
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x='Annual_Income', y='Loan_Amount', hue='Approved', 
                        alpha=0.7, palette={1: '#2ecc71', 0: '#e74c3c'} if df['Approved'].nunique() <= 2 else 'viridis')
        plt.title("Annual Income vs Loan Amount requested")
        plt.xlabel("Annual Income ($)")
        plt.ylabel("Loan Amount ($)")
        save_path = filepath_scatter or os.path.join(Config.VISUALS_DIR, 'income_loan_scatter.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @staticmethod
    def save_pairplot(df, filepath=None):
        """Generates a small multi-feature pair plot for key relationships."""
        DataVisualizer.setup_style()
        # Keep features small for rendering efficiency
        pair_cols = ['Age', 'Income', 'Loan_Amount', 'Approved']
        cols_present = [col for col in pair_cols if col in df.columns]
        
        temp_df = df[cols_present].copy()
        if 'Approved' in temp_df.columns:
            temp_df['Approved'] = temp_df['Approved'].map({1: 'Approved', 0: 'Rejected'})
            
        g = sns.pairplot(temp_df, hue='Approved', palette={'Approved': '#2ecc71', 'Rejected': '#e74c3c'})
        g.fig.suptitle("Pairwise Correlation Matrix of Core Features", y=1.02)
        
        save_path = filepath or os.path.join(Config.VISUALS_DIR, 'features_pairplot.png')
        g.savefig(save_path, bbox_inches='tight')
        plt.close()

    @staticmethod
    def save_feature_importance_plot(importance_dict, filepath=None):
        """Plots a horizontal bar chart of the top feature importance metrics."""
        DataVisualizer.setup_style()
        
        # Get top 15 features
        top_features = dict(list(importance_dict.items())[:15])
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(top_features.values()), y=list(top_features.keys()), palette="viridis")
        plt.title("Top Feature Importances (Selected Model)")
        plt.xlabel("Relative Importance Score")
        plt.ylabel("Feature Columns")
        
        save_path = filepath or os.path.join(Config.VISUALS_DIR, 'feature_importance.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @staticmethod
    def save_roc_curves(models_probs, y_test, filepath=None):
        """Generates combined ROC-AUC curve chart for model performance benchmarking."""
        DataVisualizer.setup_style()
        plt.figure(figsize=(8, 6))
        
        for model_name, y_prob in models_probs.items():
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', lw=2)
            
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=1.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Comparison')
        plt.legend(loc="lower right")
        
        save_path = filepath or os.path.join(Config.VISUALS_DIR, 'roc_comparison.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @staticmethod
    def save_confusion_matrix_plot(cm, filepath=None):
        """Saves a styled confusion matrix heatmap."""
        DataVisualizer.setup_style()
        plt.figure(figsize=(6, 5))
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=['Rejected', 'Approved'], 
                    yticklabels=['Rejected', 'Approved'])
        
        plt.title("Confusion Matrix Heatmap", pad=15)
        plt.ylabel("True Class Label")
        plt.xlabel("Predicted Class Label")
        
        save_path = filepath or os.path.join(Config.VISUALS_DIR, 'confusion_matrix.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
