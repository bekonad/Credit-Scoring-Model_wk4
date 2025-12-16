# src/data_processing.py
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

# -----------------------------
# Load Data
# -----------------------------
def load_data(file_path='data/raw/data.csv'):
    df = pd.read_csv(file_path)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

# -----------------------------
# Feature Engineering
# -----------------------------
def feature_engineering(df):
    agg_df = df.groupby('CustomerId').agg(
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        transaction_count=('TransactionId', 'count'),
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
# MANUAL WoE
# -----------------------------
def calculate_woe(df, feature, target):
    eps = 1e-6
    grouped = df.groupby(feature)[target].agg(['sum', 'count'])
    grouped['non_event'] = grouped['count'] - grouped['sum']

    event_rate = (grouped['sum'] + eps) / (df[target].sum() + eps)
    non_event_rate = (grouped['non_event'] + eps) / ((df[target] == 0).sum() + eps)

    grouped['woe'] = np.log(event_rate / non_event_rate)
    return df[feature].map(grouped['woe'])

def apply_woe(df, categorical_cols, target):
    df_woe = df.copy()
    for col in categorical_cols:
        df_woe[col] = calculate_woe(df_woe, col, target)
    return df_woe

# -----------------------------
# Preprocessing Pipeline
# -----------------------------
def build_preprocessor(numeric_features):
    return ColumnTransformer(
        transformers=[
            ('num',
             Pipeline([
                 ('imputer', SimpleImputer(strategy='median')),
                 ('scaler', StandardScaler())
             ]),
             numeric_features)
        ]
    )

# -----------------------------
# Full Processing Function
# -----------------------------
def process_data(
    input_path='data/raw/data.csv',
    output_path='data/processed/processed.csv'
):
    raw_df = load_data(input_path)

    features = feature_engineering(raw_df)
    rfm = calculate_rfm(raw_df)
    target = create_proxy_target(rfm)

    df = features.merge(target, on='CustomerId')

    categorical_cols = [
        'CountryCode', 'CurrencyCode',
        'ChannelId', 'ProductCategory',
        'ProviderId', 'PricingStrategy'
    ]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    if categorical_cols:
        df = apply_woe(df, categorical_cols, 'is_high_risk')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    numeric_features = [
        'total_amount', 'avg_amount',
        'transaction_count', 'std_amount',
        'avg_hour', 'avg_day'
    ]

    preprocessor = build_preprocessor(numeric_features)
    return df, preprocessor

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    df, pipeline = process_data()
    print("✅ Processed data saved:", df.shape)
    print(df.head())