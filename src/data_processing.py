# src/data_processing.py
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from datetime import datetime
# -----------------------------
# Load Data
# -----------------------------
def load_data(file_path='data/raw/data.csv'):
    df = pd.read_csv(file_path)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

# -----------------------------
# Aggregate & Time-based Features
# -----------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate features per customer
    agg_df = df.groupby('CustomerId').agg(
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        transaction_count=('TransactionId', 'count'),
        std_amount=('Amount', 'std')
    ).reset_index()
    agg_df['std_amount'] = agg_df['std_amount'].fillna(0)

    # Extract time-based features
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_year'] = df['TransactionStartTime'].dt.year

    # Merge aggregates back
    df = df.merge(agg_df, on='CustomerId', how='left')
    return df

# -----------------------------
# Weight of Evidence (WoE) Transformation
# -----------------------------
def calculate_woe(df: pd.DataFrame, col: str, target: str):
    eps = 0.0001
    grouped = df.groupby(col)[target].agg(['sum','count'])
    grouped['non_event'] = grouped['count'] - grouped['sum']
    grouped['event_rate'] = (grouped['sum'] + eps) / (df[target].sum() + eps)
    grouped['non_event_rate'] = (grouped['non_event'] + eps) / ((df[target] == 0).sum() + eps)
    grouped['woe'] = np.log(grouped['event_rate'] / grouped['non_event_rate'])
    return df[col].map(grouped['woe'])

def apply_woe(df: pd.DataFrame, categorical_cols: list, target_col='is_high_risk'):
    df_woe = df.copy()
    for col in categorical_cols:
        if col in df_woe.columns:
            df_woe[col] = calculate_woe(df_woe, col, target_col)
    return df_woe

# -----------------------------
# Preprocessing Pipeline
# -----------------------------
def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features = [col for col in numeric_features if col not in ['CustomerId', 'TransactionId', 'BatchId', 'SubscriptionId']]

    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

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
# Full Processing Function
# -----------------------------
def process_data(input_path='data/raw/data.csv', output_path='data/processed/processed.csv', target_col=None):
    df = load_data(input_path)
    df = feature_engineering(df)

    # Apply WoE if target_col is provided
    categorical_cols = ['CountryCode', 'CurrencyCode', 'ChannelId', 'ProductCategory', 'ProviderId', 'PricingStrategy']
    categorical_cols = [col for col in categorical_cols if col in df.columns and target_col is not None]
    if target_col and categorical_cols:
        df = apply_woe(df, categorical_cols, target_col=target_col)

    # Save processed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(df)
    return df, preprocessor

# -----------------------------
# Script Entry Point
# -----------------------------
if __name__ == "__main__":
    processed_df, pipeline = process_data(
        input_path='data/raw/data.csv',
        output_path='data/processed/processed.csv',
        target_col=None  # Set 'is_high_risk' later in Task 4
    )
    print("Processed data saved at 'data/processed/processed.csv'. Shape:", processed_df.shape)
    print(processed_df.head())