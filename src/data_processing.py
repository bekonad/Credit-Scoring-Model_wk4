# src/data_processing.py
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from xverse.transformer import WOE
from datetime import datetime

# -----------------------------
# Load Data
# -----------------------------
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

# -----------------------------
# Feature Aggregation
# -----------------------------
def create_aggregates(df):
    agg_df = df.groupby('CustomerId').agg(
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        transaction_count=('Amount', 'count'),
        std_amount=('Amount', 'std'),
        avg_hour=('TransactionStartTime', lambda x: x.dt.hour.mean()),
        avg_day=('TransactionStartTime', lambda x: x.dt.day.mean())
    ).reset_index()
    agg_df['std_amount'] = agg_df['std_amount'].fillna(0)
    return agg_df

# -----------------------------
# RFM Metrics
# -----------------------------
def calculate_rfm(df):
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Amount', 'sum')
    ).reset_index()
    rfm['Monetary'] = rfm['Monetary'].abs()
    return rfm

# -----------------------------
# Proxy Target Creation
# -----------------------------
def create_proxy_target(rfm):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)
    cluster_means = rfm.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_means['Recency'].idxmax()
    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)
    return rfm[['CustomerId', 'is_high_risk']]

# -----------------------------
# WoE Transformation for Categorical Features
# -----------------------------
def apply_woe(df, categorical_cols, target_col='is_high_risk'):
    df_woe = df.copy()
    woe_transformer = WOE(cols=categorical_cols, target=target_col)
    woe_transformer.fit(df_woe, df_woe[target_col])
    df_woe[categorical_cols] = woe_transformer.transform(df_woe)
    return df_woe, woe_transformer

# -----------------------------
# Preprocessing Pipeline
# -----------------------------
def preprocess_pipeline(numeric_features):
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[('num', num_transformer, numeric_features)]
    )
    return preprocessor

# -----------------------------
# Full Processing Function
# -----------------------------
def process_data(input_path, output_path):
    df = load_data(input_path)

    # Aggregate numeric and time features
    agg_df = create_aggregates(df)

    # Compute RFM metrics and proxy target
    rfm = calculate_rfm(df)
    rfm_target = create_proxy_target(rfm)

    # Merge numeric features with target
    processed_df = agg_df.merge(rfm_target, on='CustomerId')

    # Identify categorical features for WoE
    categorical_cols = ['CountryCode', 'CurrencyCode', 'ChannelId', 'ProductCategory', 'ProviderId', 'PricingStrategy']
    categorical_cols = [col for col in categorical_cols if col in processed_df.columns]

    # Apply WoE transformation
    if categorical_cols:
        processed_df, woe_transformer = apply_woe(processed_df, categorical_cols, target_col='is_high_risk')

    # Save processed dataset
    processed_df.to_csv(output_path, index=False)

    # Define preprocessing pipeline for numeric columns
    numeric_features = ['total_amount', 'avg_amount', 'transaction_count', 'std_amount', 'avg_hour', 'avg_day']
    preprocessor = preprocess_pipeline(numeric_features)

    return processed_df, preprocessor

# -----------------------------
# Script Entry Point
# -----------------------------
if __name__ == "__main__":
    processed_data, pipeline = process_data(
        input_path='data/raw/data.csv',
        output_path='data/processed/processed.csv'
    )
    print("Processed data saved. Shape:", processed_data.shape)
