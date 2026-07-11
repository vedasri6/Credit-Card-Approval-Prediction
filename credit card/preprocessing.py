import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from config import Config

class DataPreprocessor:
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.fitted = False

    def handle_outliers(self, df, columns, factor=1.5):
        """Caps numeric outliers to the 1.5 * IQR boundaries."""
        df_copy = df.copy()
        for col in columns:
            if col in df_copy.columns:
                q1 = df_copy[col].quantile(0.25)
                q3 = df_copy[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - factor * iqr
                upper_bound = q3 + factor * iqr
                df_copy[col] = np.clip(df_copy[col], lower_bound, upper_bound)
        return df_copy

    def fit(self, df):
        """Fits imputers, scaler, and encoder on the training DataFrame."""
        # Split features
        num_cols = [col for col in Config.NUMERIC_FEATURES if col in df.columns]
        cat_cols = [col for col in Config.CATEGORICAL_FEATURES if col in df.columns]
        
        # Fit Numeric Imputer
        if num_cols:
            self.num_imputer.fit(df[num_cols])
            
        # Fit Categorical Imputer
        if cat_cols:
            self.cat_imputer.fit(df[cat_cols])
            
        # Transform temporarily to fit Scaler and Encoder
        imputed_num = self.num_imputer.transform(df[num_cols]) if num_cols else np.empty((len(df), 0))
        imputed_cat = self.cat_imputer.transform(df[cat_cols]) if cat_cols else np.empty((len(df), 0))
        
        if num_cols:
            self.scaler.fit(imputed_num)
        if cat_cols:
            self.encoder.fit(imputed_cat)
            
        self.fitted = True
        return self

    def transform(self, df):
        """Transforms a DataFrame using the fitted preprocessors."""
        if not self.fitted:
            raise ValueError("DataPreprocessor must be fitted before transforming data.")
            
        df_copy = df.copy()
        
        num_cols = [col for col in Config.NUMERIC_FEATURES if col in df_copy.columns]
        cat_cols = [col for col in Config.CATEGORICAL_FEATURES if col in df_copy.columns]
        
        # Impute
        if num_cols:
            df_copy[num_cols] = self.num_imputer.transform(df_copy[num_cols])
            # Cap outliers
            df_copy = self.handle_outliers(df_copy, num_cols)
            # Scale
            scaled_num = self.scaler.transform(df_copy[num_cols])
        else:
            scaled_num = np.empty((len(df_copy), 0))
            
        if cat_cols:
            df_copy[cat_cols] = self.cat_imputer.transform(df_copy[cat_cols])
            # Encode
            encoded_cat = self.encoder.transform(df_copy[cat_cols])
        else:
            encoded_cat = np.empty((len(df_copy), 0))
            
        # Combine processed numeric and categorical features
        processed_features = np.hstack((scaled_num, encoded_cat))
        
        # Build feature names list
        feature_names = []
        if num_cols:
            feature_names.extend(num_cols)
        if cat_cols:
            cat_feature_names = self.encoder.get_feature_names_out(cat_cols)
            feature_names.extend(cat_feature_names)
            
        return pd.DataFrame(processed_features, columns=feature_names)

    def fit_transform(self, df):
        """Convenience method to fit and transform."""
        self.fit(df)
        return self.transform(df)
