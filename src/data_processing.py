import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from datetime import datetime

# -----------------------------
# Load raw data
# -----------------------------
def load_data(file_path='data/raw/data.csv'):
    df = pd.read_csv(file_path)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

# -----------------------------
# Feature Engineering
# -----------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate features per customer
    agg_df = df.groupby('CustomerId').agg(
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        transaction_count=('TransactionId', 'count'),
        std_amount=('Amount', 'std')
    ).reset_index()

    # Extract time-based features
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_year'] = df['TransactionStartTime'].dt.year

    # Merge aggregates back
    df = df.merge(agg_df, on='CustomerId', how='left')
    return df

# -----------------------------
# Build preprocessing pipeline
# -----------------------------
def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    # Identify numerical and categorical columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features = [col for col in numeric_features if col not in ['CustomerId', 'TransactionId', 'BatchId', 'SubscriptionId']]
    
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

# -----------------------------
# Full preprocessing pipeline
# -----------------------------
def preprocess_data(file_path='data/raw/data.csv'):
    df = load_data(file_path)
    df = feature_engineering(df)
    preprocessor = build_preprocessor(df)
    return df, preprocessor

# -----------------------------
# Example usage
# -----------------------------
if __name__ == '__main__':
    df, preprocessor = preprocess_data()
    print("Processed dataframe sample:")
    print(df.head())