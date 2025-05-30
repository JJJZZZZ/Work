# %% [markdown]
# # Load  Data

# %%
# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import featuretools as ft
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import datetime as dt
import time
import trino
import getpass

# %%
# ---- Execution Section: No Action Required ----

# Configures pandas to display all rows and columns.

# Set the maximum number of displayed rows to None (show all rows)
pd.set_option("display.max_rows", None)

# Set the maximum number of displayed columns to None (show all columns)
pd.set_option("display.max_columns", None)

# Disable truncation of DataFrame string representations
pd.set_option("display.width", None)

# Disable truncation of column contents
pd.set_option("display.max_colwidth", None)

# %% [markdown]
# <div style="background-color:#ccffcc; padding:20px; border-radius:10px; text-align:left;">
#     <h1 style="margin:0;">STEP 2: Read and Process Data</h1>
# </div>

# %%
"""
This script establishes a connection to a Presto database via the Trino Python client,
executes a SQL query to fetch data, and then structures the fetched data into a pandas DataFrame.
The DataFrame is organized with appropriate column names, making the data ready for analysis.
"""

import trino
import pandas as pd
import datetime
import getpass
import os
import keyring
import json
from pathlib import Path
from datetime import datetime, timedelta

# Function to get credentials from keyring or config file
def get_credentials():
    username = "jiawei.zhou@mongodb.com"
    # Try to get password from keyring
    password = keyring.get_password("trino_db", username)
    
    # If not in keyring, check for config file
    if not password:
        config_path = Path.home() / ".trino_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
                password = config.get("password")
    
    # If still no password, prompt user and save it
    if not password:
        password = getpass.getpass(f"Enter password for {username}: ")
        # Save to keyring
        keyring.set_password("trino_db", username, password)
        # Also save to config file as backup
        with open(Path.home() / ".trino_config.json", "w") as f:
            json.dump({"username": username, "password": password}, f)
        print("Credentials saved for future use")
    
    return username, password

# Get credentials
username, password = get_credentials()

# Prompt for model score date
model_score_date = input("Enter the date for model scoring (YYYY-MM-DD): ")
end_date_obj = datetime.strptime(model_score_date, '%Y-%m-%d')
start_date_obj = end_date_obj - timedelta(days=30)
start_date = start_date_obj.strftime('%Y-%m-%d')

# Establishing the connection
conn = trino.dbapi.connect(
    host='presto-gateway.corp.mongodb.com',
    port=443,
    user=username,
    catalog='awsdatacatalog',
    http_scheme='https',
    auth=trino.auth.BasicAuthentication(username, password),
)

# Executing the query
cur = conn.cursor()

cur.execute(f"""
SELECT *
FROM awsdatacatalog.ns__analytics_postprocessing.flywheel_db_cluster_daily_metrics
WHERE ds BETWEEN DATE '{start_date}' AND DATE '{model_score_date}'
AND cluster_mrr > 0 
AND data_size_total_gb > 0
AND instance_size IS NOT NULL
AND cluster_mdb_major_version IS NOT NULL
AND instance_size != 'SERVERLESS'
""")

# Extracting the column names
columns = [desc[0] for desc in cur.description]

# Fetching the rows
rows = cur.fetchall()

# Creating the DataFrame with the fetched rows and column names
df_backup = pd.DataFrame(rows, columns=columns)

# Displaying the first few rows of the DataFrame
df_backup.head()


# %% [markdown]
# <div style="background-color:#ccffcc; padding:20px; border-radius:10px; text-align:left;">
#     <h1 style="margin:0;">STEP 3: Data Manipulation</h1>
# </div>

# %%
# Create a copy of the original DataFrame
df = df_backup.copy()

# %%
# Convert 'ds' column to datetime type
df["ds"] = pd.to_datetime(df["ds"])

# %%
# Count number of rows per date
date_counts = df.groupby('ds').size()

# Display the counts
print("Number of rows per date:")
print(date_counts)

# Display total number of unique dates
print(f"\nTotal number of unique dates: {len(date_counts)}")


# %%
# Extract numeric part from instance_size and convert to numeric
df["instance_size_numeric"] = df["instance_size"].str.extract(r"(\d+)").astype(float)

# Convert cluster_mdb_major_version to numeric
df["cluster_mdb_major_version"] = pd.to_numeric(
    df["cluster_mdb_major_version"], errors="coerce"
)

# Display the first few rows to verify the changes
print(
    df[["instance_size", "instance_size_numeric", "cluster_mdb_major_version"]].head()
)

# %%
# Purpose: Calculate rolling averages and percentage changes for key metrics to smooth out daily fluctuations
# and identify growth trends

# Sort chronologically by cluster before calculations
df = df.sort_values(["cluster_id", "ds"])

# Define metrics to analyze
columns_to_smooth = [
    "cluster_mrr",
    "reads_writes_per_second_avg", 
    "data_size_total_gb",
    "documents_total",
    "database_count",
    "collection_count",
    "total_accesses", 
    "total_indexes",
    "auditlog",
    "agg_total_calls",
    "agg_set_calls",
    "agg_match_calls", 
    "agg_group_calls",
    "agg_project_calls",
    "agg_limit_calls",
    "agg_sort_calls",
    "agg_unwind_calls",
    "agg_add_fields_calls",
    "agg_lookup_calls",
    "agg_count_calls",
    "agg_skip_calls",
    "cluster_analytics_node_count",
    "adminapi_directapi_usage",
    "agg_change_stream_calls",
    "changestream_events",
    "total_hosts_transactions_transactionscollectionwritecount",
    "oa_1d_usage_filesize",
    "text_indexes",
    "text_active_1d_usage", 
    "ts_collection_count",
    "trigger_events",
    "vector_indexes",
    "vector_active_1d_usage",
]

# Calculate 30-day rolling averages for each metric by cluster
for col in columns_to_smooth:
    smoothed_col = f"smoothed_{col}"
    df[smoothed_col] = df.groupby("cluster_id")[col].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )

# Calculate 30-day percentage changes using smoothed values
for col in columns_to_smooth:
    smoothed_col = f"smoothed_{col}"
    df[f"{smoothed_col}_30d_pct_change"] = df.groupby("cluster_id")[
        smoothed_col
    ].transform(lambda x: x.pct_change(periods=30, fill_method=None))

# %%
# Purpose: Calculate relative usage percentiles for each smoothed metric within instance size groups
# to normalize usage metrics across different instance sizes

# Filter for the predefined date
df_filtered = df[df['ds'] == end_date_obj]

# Get columns with smoothed metrics and fill missing values with 0
smoothed_columns = [col for col in df_filtered.columns if col.startswith("smoothed")] 
df_filtered[smoothed_columns] = df_filtered[smoothed_columns].fillna(0)

def percentile_rank(series):
    """Convert values to percentile ranks (0-1) within their group"""
    return series.rank(pct=True)

# Pre-calculate all percentile ranks at once to avoid DataFrame fragmentation
percentile_ranks = {}
for col in smoothed_columns:
    percentile_ranks[f"{col}_percentile_by_instance"] = df_filtered.groupby("instance_size")[col].transform(percentile_rank)

# Add all percentile columns at once using concat
df_filtered = pd.concat([df_filtered, pd.DataFrame(percentile_ranks)], axis=1)

# %%
df_filtered.head()

# %% [markdown]
# ### Core Product Engagement

# %%
# Define the core metrics we want to analyze (using percentile by instance metrics)
features = [
    "smoothed_reads_writes_per_second_avg_percentile_by_instance",
    "smoothed_data_size_total_gb_percentile_by_instance",
    "smoothed_documents_total_percentile_by_instance",
    "smoothed_database_count_percentile_by_instance",
    "smoothed_collection_count_percentile_by_instance",
]

# Import joblib
import joblib

# Load the trained model
pipeline = joblib.load('core_product_engagement_model.joblib')

# Apply model to get prediction scores
df_filtered["core_product_engagement_score"] = pipeline.predict_proba(df_filtered[features])[:, 1]

# %% [markdown]
# ### Core Product Trends

# %%
# Define the core metrics we want to analyze (using percentile by instance metrics)
features = [
    "smoothed_reads_writes_per_second_avg_30d_pct_change_percentile_by_instance", 
    "smoothed_data_size_total_gb_30d_pct_change_percentile_by_instance",
    "smoothed_documents_total_30d_pct_change_percentile_by_instance",
    "smoothed_database_count_30d_pct_change_percentile_by_instance",
    "smoothed_collection_count_30d_pct_change_percentile_by_instance",
]

# Load the trained model
pipeline = joblib.load('core_product_trend_model.joblib')

# Apply model to get prediction scores 
df_filtered["core_product_trend_score"] = pipeline.predict_proba(df_filtered[features])[:, 1]


# %% [markdown]
# ### Additional Product Engagement
# 

# %%
df_filtered.head()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Purpose: Analyze relationship between product feature usage and MRR growth
# Analyze various product metrics normalized by instance size to identify growth indicators

# Key feature categories:
# - Basic usage: accesses, indexes
# - Aggregation pipeline usage: total calls, match, group, etc.
# - Advanced features: analytics, transactions, search, audit logs etc.
features = [
    # Advanced index usage metrics
    "smoothed_total_accesses_percentile_by_instance",
    "smoothed_total_indexes_percentile_by_instance",
    
    # Aggregation pipeline metrics  
    "smoothed_agg_total_calls_percentile_by_instance",
    "smoothed_agg_set_calls_percentile_by_instance",
    "smoothed_agg_match_calls_percentile_by_instance", 
    "smoothed_agg_group_calls_percentile_by_instance",
    "smoothed_agg_project_calls_percentile_by_instance",
    "smoothed_agg_limit_calls_percentile_by_instance",
    "smoothed_agg_sort_calls_percentile_by_instance",
    "smoothed_agg_unwind_calls_percentile_by_instance",
    "smoothed_agg_add_fields_calls_percentile_by_instance",
    "smoothed_agg_lookup_calls_percentile_by_instance",
    "smoothed_agg_count_calls_percentile_by_instance",
    "smoothed_agg_skip_calls_percentile_by_instance",
    
    # Advanced feature usage
    "smoothed_cluster_analytics_node_count_percentile_by_instance",
    "smoothed_adminapi_directapi_usage_percentile_by_instance", 
    "smoothed_agg_change_stream_calls_percentile_by_instance",
    "smoothed_changestream_events_percentile_by_instance",
    "smoothed_total_hosts_transactions_transactionscollectionwritecount_percentile_by_instance",
    "smoothed_oa_1d_usage_filesize_percentile_by_instance",
    "smoothed_text_indexes_percentile_by_instance",
    "smoothed_text_active_1d_usage_percentile_by_instance",
    "smoothed_ts_collection_count_percentile_by_instance",
    "smoothed_trigger_events_percentile_by_instance",
    "smoothed_vector_indexes_percentile_by_instance",
    "smoothed_vector_active_1d_usage_percentile_by_instance",
    "smoothed_auditlog_percentile_by_instance",
]


# Load and apply the additional feature engagement model
pipeline = joblib.load('additional_feature_engagement_model.joblib')
df_filtered["additional_feature_engagement_score"] = pipeline.predict_proba(df_filtered[features])[:, 1]

# %% [markdown]
# ### Additional Feature Trends

# %%
# Purpose: Analyze how changes in feature usage over 30 days correlate with MRR growth
# by comparing percentile metrics vs 6-month MRR growth threshold

# Define feature usage trend metrics (30-day percentage changes)
features = [
    # Advanced index usage metrics
    "smoothed_total_accesses_30d_pct_change_percentile_by_instance",
    "smoothed_total_indexes_30d_pct_change_percentile_by_instance",
    
    # Aggregation pipeline usage
    "smoothed_agg_total_calls_30d_pct_change_percentile_by_instance",
    "smoothed_agg_set_calls_30d_pct_change_percentile_by_instance", 
    "smoothed_agg_match_calls_30d_pct_change_percentile_by_instance",
    "smoothed_agg_group_calls_30d_pct_change_percentile_by_instance",
    "smoothed_agg_project_calls_30d_pct_change_percentile_by_instance",
    "smoothed_agg_limit_calls_30d_pct_change_percentile_by_instance",
    "smoothed_agg_sort_calls_30d_pct_change_percentile_by_instance",
    "smoothed_agg_unwind_calls_30d_pct_change_percentile_by_instance",
    "smoothed_agg_add_fields_calls_30d_pct_change_percentile_by_instance",
    "smoothed_agg_lookup_calls_30d_pct_change_percentile_by_instance",
    "smoothed_agg_count_calls_30d_pct_change_percentile_by_instance",
    "smoothed_agg_skip_calls_30d_pct_change_percentile_by_instance",
    
    # Advanced feature usage
    "smoothed_cluster_analytics_node_count_30d_pct_change_percentile_by_instance",
    "smoothed_adminapi_directapi_usage_30d_pct_change_percentile_by_instance",
    "smoothed_agg_change_stream_calls_30d_pct_change_percentile_by_instance", 
    "smoothed_changestream_events_30d_pct_change_percentile_by_instance",
    "smoothed_total_hosts_transactions_transactionscollectionwritecount_30d_pct_change_percentile_by_instance",
    "smoothed_oa_1d_usage_filesize_30d_pct_change_percentile_by_instance",
    "smoothed_text_indexes_30d_pct_change_percentile_by_instance",
    "smoothed_text_active_1d_usage_30d_pct_change_percentile_by_instance",
    "smoothed_ts_collection_count_30d_pct_change_percentile_by_instance",
    "smoothed_trigger_events_30d_pct_change_percentile_by_instance",
    "smoothed_vector_indexes_30d_pct_change_percentile_by_instance",
    "smoothed_vector_active_1d_usage_30d_pct_change_percentile_by_instance",
    "smoothed_auditlog_30d_pct_change_percentile_by_instance"
]

# Load and apply the additional feature trend model
pipeline = joblib.load('additional_feature_trend_model.joblib')
df_filtered["additional_feature_trend_score"] = pipeline.predict_proba(df_filtered[features])[:, 1]

# %% [markdown]
# ### Customer Demographics and Account Characteristics

# %%
# Define features
features = [
    "cluster_age_month",
    "cluster_mdb_major_version",
    "is_backup_selected", 
    "is_auto_expand_storage",
    "is_auto_scaling_compute_enabled",
    "is_auto_scaling_compute_scaledown_enabled",
    "instance_size_numeric",
]

# Load and apply the demo score model
pipeline = joblib.load('demo_score_model.joblib')
df_filtered["demo_score"] = pipeline.predict_proba(df_filtered[features])[:, 1]

# %% [markdown]
# ### Final Model

# %%
df_filtered.head()

# %%
# Select features
features = [
    "demo_score",
    "core_product_engagement_score",
    "core_product_trend_score", 
    "additional_feature_engagement_score",
    "additional_feature_trend_score"
]

# Load the saved model
pipeline = joblib.load('growth_score_model.joblib')

# Apply model to get growth score
df_filtered["cluster_growth_score"] = pipeline.predict_proba(df_filtered[features])[:, 1]

# %%
# Calculate deciles (1-10) for all scores
df_filtered["demo_score_decile"] = pd.qcut(df_filtered["demo_score"], q=10, labels=False, duplicates='drop') + 1
df_filtered["core_product_engagement_score_decile"] = pd.qcut(df_filtered["core_product_engagement_score"], q=10, labels=False, duplicates='drop') + 1 
df_filtered["core_product_trend_score_decile"] = pd.qcut(df_filtered["core_product_trend_score"], q=10, labels=False, duplicates='drop') + 1
df_filtered["additional_feature_engagement_score_decile"] = pd.qcut(df_filtered["additional_feature_engagement_score"], q=10, labels=False, duplicates='drop') + 1
df_filtered["additional_feature_trend_score_decile"] = pd.qcut(df_filtered["additional_feature_trend_score"], q=10, labels=False, duplicates='drop') + 1
df_filtered["cluster_growth_decile"] = pd.qcut(df_filtered["cluster_growth_score"], q=10, labels=False, duplicates='drop') + 1

# %%
# Plot growth score distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot raw growth score distribution
sns.histplot(df_filtered["cluster_growth_score"], ax=ax1)
ax1.set_title("Distribution of Cluster Growth Score")
ax1.set_xlabel("Cluster Growth Score")
ax1.set_ylabel("Count")

# Plot 1-10 score distribution 
sns.histplot(df_filtered["cluster_growth_decile"], ax=ax2)
ax2.set_title("Distribution of Cluster Growth Score (1-10 Scale)")
ax2.set_xlabel("Cluster Growth Score (1-10)")
ax2.set_ylabel("Count")

plt.tight_layout()
plt.show()

# %%
df_filtered.head()

# %% [markdown]
# # Aggregation to Account

# %%
# Calculate weighted growth score using raw growth_score and smoothed MRR
df_filtered["weighted_score"] = df_filtered["cluster_growth_score"] * df_filtered["smoothed_cluster_mrr"]

# Group by ultimate parent account and calculate weighted average
account_scores = df_filtered.groupby("ultimate_parent_account_id").agg({
    "weighted_score": "sum",
    "smoothed_cluster_mrr": "sum",
    "ds": "max"
}).reset_index()

# Calculate account growth score for each account
account_scores["account_growth_score"] = account_scores["weighted_score"] / account_scores["smoothed_cluster_mrr"]

# Assign decile scores (1-10) based on account growth score
account_scores["account_growth_decile"] = pd.qcut(account_scores["account_growth_score"], q=10, labels=range(1,11)).astype(float)

# Add processed date column with today's date
account_scores["processed_date"] = pd.Timestamp.today().strftime("%Y-%m-%d")

# %%
account_scores.head()

# %%
# Plot weighted average score distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot weighted average score distribution
sns.histplot(account_scores["account_growth_score"], ax=ax1)
ax1.set_title("Distribution of Account Growth Score")
ax1.set_xlabel("Account Growth Score") 
ax1.set_ylabel("Count")

# Plot account growth decile distribution
sns.histplot(account_scores["account_growth_decile"], ax=ax2)
ax2.set_title("Distribution of Account Growth Score (1-10 Scale)")
ax2.set_xlabel("Account Growth Score (1-10)")
ax2.set_ylabel("Count")

plt.tight_layout()
plt.show()


# %% [markdown]
# # Insert Data
# 

# %%
# Load landing zone list with only account name and id columns
landing_zone_list = pd.read_csv("Enterprise Flywheel - Account Activity Tracker - List for Analytics.csv", 
                               usecols=['Account Name', 'Account ID'])

# Clean up column names and convert to lowercase
landing_zone_list.columns = landing_zone_list.columns.str.strip().str.lower()

# Rename columns to more descriptive names
column_mapping = {
    'account name': 'account_name',
    'account id': 'account_id'
}
landing_zone_list = landing_zone_list.rename(columns=column_mapping)

# %% [markdown]
# ### Accout Score

# %%
# Create table
conn = trino.dbapi.connect(
    host='presto-gateway.corp.mongodb.com',
    port=443,
    user='jiawei.zhou@mongodb.com', 
    catalog='awsdatacatalog',
    http_scheme='https',
    auth=trino.auth.BasicAuthentication("jiawei.zhou@mongodb.com", password)
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS awsdatacatalog.product_analytics_internal.account_score_enterprise_flywheel (
    ultimate_parent_account_id VARCHAR,
    weighted_score DOUBLE,
    smoothed_cluster_mrr DOUBLE,
    account_growth_score DOUBLE,
    account_growth_decile DOUBLE,
    processed_date DATE,
    ds DATE
)
WITH (
    format = 'PARQUET',
    partitioned_by = ARRAY['ds']
)
""")

conn.commit()
cursor.close()
conn.close()

print("Successfully created account_score_enterprise_flywheel table")

# %%
account_scores.head()

# %%
landing_zone_list.head()

# %%
# Insert data in batches
BATCH_SIZE = 100

# Convert timestamp columns to date before inserting
account_scores['ds'] = pd.to_datetime(account_scores['ds']).dt.date
account_scores['processed_date'] = pd.to_datetime(account_scores['processed_date']).dt.date

# Filter account_scores to only include accounts from landing_zone_list
valid_accounts = landing_zone_list['account_id'].unique()
filtered_scores = account_scores[account_scores['ultimate_parent_account_id'].isin(valid_accounts)]

# Calculate number of batches
num_batches = len(filtered_scores) // BATCH_SIZE + (1 if len(filtered_scores) % BATCH_SIZE else 0)

total_inserted = 0

for batch in range(num_batches):
    start_idx = batch * BATCH_SIZE
    end_idx = min((batch + 1) * BATCH_SIZE, len(filtered_scores))
    
    batch_df = filtered_scores.iloc[start_idx:end_idx]
    
    conn = trino.dbapi.connect(
        host='presto-gateway.corp.mongodb.com',
        port=443,
        user='jiawei.zhou@mongodb.com', 
        catalog='awsdatacatalog',
        http_scheme='https',
        auth=trino.auth.BasicAuthentication("jiawei.zhou@mongodb.com", password)
    )
    
    cursor = conn.cursor()
    
    # Build batch insert query
    values_list = []
    for _, row in batch_df.iterrows():
        values_list.append(f"""(
            '{row['ultimate_parent_account_id']}',
            {row['weighted_score']:.2f},
            {row['smoothed_cluster_mrr']:.2f},
            {row['account_growth_score']:.2f},
            {row['account_growth_decile']:.2f},
            DATE '{row['processed_date']}',
            DATE '{row['ds']}'
        )""")
    
    values_str = ",\n".join(values_list)
    
    insert_query = f"""
    INSERT INTO product_analytics_internal.account_score_enterprise_flywheel (
        ultimate_parent_account_id,
        weighted_score,
        smoothed_cluster_mrr,
        account_growth_score,
        account_growth_decile,
        processed_date,
        ds
    ) VALUES {values_str}
    """
    
    cursor.execute(insert_query)
    conn.commit()
    cursor.close()
    conn.close()
    
    total_inserted += len(batch_df)
    print(f"Inserted batch {batch + 1}/{num_batches} ({total_inserted}/{len(filtered_scores)} rows)")

print(f"Successfully inserted all {total_inserted} rows of data")

# %%
# Get list of account IDs from landing zone that are not in account score
landing_zone_df = pd.read_csv('Enterprise Flywheel - Account Activity Tracker - List for Analytics.csv')
account_score_df = filtered_scores[['ultimate_parent_account_id']].drop_duplicates()

missing_accounts = landing_zone_df[~landing_zone_df['Account ID'].isin(account_score_df['ultimate_parent_account_id'])]

missing_accounts

# %% [markdown]
# ### Cluster Score

# %%
# Create table
conn = trino.dbapi.connect(
    host='presto-gateway.corp.mongodb.com',
    port=443,
    user='jiawei.zhou@mongodb.com', 
    catalog='awsdatacatalog',
    http_scheme='https',
    auth=trino.auth.BasicAuthentication("jiawei.zhou@mongodb.com", password)
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS awsdatacatalog.product_analytics_internal.cluster_score_enterprise_flywheel (
    cluster_id VARCHAR,
    group_id VARCHAR,
    org_id VARCHAR,
    account_id VARCHAR,
    ultimate_parent_account_id VARCHAR,
    cluster_age_month INTEGER,
    cluster_analytics_node_count INTEGER,
    cluster_mdb_major_version DOUBLE,
    instance_size VARCHAR,
    is_auto_expand_storage BOOLEAN,
    is_auto_scaling_compute_enabled BOOLEAN,
    is_auto_scaling_compute_scaledown_enabled BOOLEAN,
    is_backup_selected BOOLEAN,
    is_sharding BOOLEAN,
    region_count INTEGER,
    shard_count INTEGER,
    total_accesses DOUBLE,
    total_indexes DOUBLE,
    agg_add_fields_calls DOUBLE,
    agg_change_stream_calls DOUBLE,
    agg_count_calls DOUBLE,
    agg_group_calls DOUBLE,
    agg_limit_calls DOUBLE,
    agg_lookup_calls DOUBLE,
    agg_match_calls DOUBLE,
    agg_project_calls DOUBLE,
    agg_set_calls DOUBLE,
    agg_skip_calls DOUBLE,
    agg_sort_calls DOUBLE,
    agg_unwind_calls DOUBLE,
    agg_total_calls DOUBLE,
    changestream_events DOUBLE,
    data_size_total_gb DOUBLE,
    documents_total DOUBLE,
    reads_writes_per_second_avg DOUBLE,
    collection_count DOUBLE,
    database_count DOUBLE,
    cluster_mrr DOUBLE,
    oa_1d_usage_filesize DOUBLE,
    trigger_events DOUBLE,
    text_active_1d_usage DOUBLE,
    text_indexes DOUBLE,
    vector_active_1d_usage DOUBLE,
    vector_indexes DOUBLE,
    ts_collection_count DOUBLE,
    total_hosts_transactions_transactionscollectionwritecount DOUBLE,
    auditlog VARCHAR,
    adminapi_directapi_usage DOUBLE,
    instance_size_numeric DOUBLE,
    smoothed_cluster_mrr DOUBLE,
    smoothed_reads_writes_per_second_avg DOUBLE,
    smoothed_data_size_total_gb DOUBLE,
    smoothed_documents_total DOUBLE,
    smoothed_database_count DOUBLE,
    smoothed_collection_count DOUBLE,
    smoothed_total_accesses DOUBLE,
    smoothed_total_indexes DOUBLE,
    smoothed_auditlog DOUBLE,
    smoothed_agg_total_calls DOUBLE,
    smoothed_agg_set_calls DOUBLE,
    smoothed_agg_match_calls DOUBLE,
    smoothed_agg_group_calls DOUBLE,
    smoothed_agg_project_calls DOUBLE,
    smoothed_agg_limit_calls DOUBLE,
    smoothed_agg_sort_calls DOUBLE,
    smoothed_agg_unwind_calls DOUBLE,
    smoothed_agg_add_fields_calls DOUBLE,
    smoothed_agg_lookup_calls DOUBLE,
    smoothed_agg_count_calls DOUBLE,
    smoothed_agg_skip_calls DOUBLE,
    smoothed_cluster_analytics_node_count DOUBLE,
    smoothed_adminapi_directapi_usage DOUBLE,
    smoothed_agg_change_stream_calls DOUBLE,
    smoothed_changestream_events DOUBLE,
    smoothed_total_hosts_transactions_transactionscollectionwritecount DOUBLE,
    smoothed_oa_1d_usage_filesize DOUBLE,
    smoothed_text_indexes DOUBLE,
    smoothed_text_active_1d_usage DOUBLE,
    smoothed_ts_collection_count DOUBLE,
    smoothed_trigger_events DOUBLE,
    smoothed_vector_indexes DOUBLE,
    smoothed_vector_active_1d_usage DOUBLE,
    smoothed_cluster_mrr_30d_pct_change DOUBLE,
    smoothed_reads_writes_per_second_avg_30d_pct_change DOUBLE,
    smoothed_data_size_total_gb_30d_pct_change DOUBLE,
    smoothed_documents_total_30d_pct_change DOUBLE,
    smoothed_database_count_30d_pct_change DOUBLE,
    smoothed_collection_count_30d_pct_change DOUBLE,
    smoothed_total_accesses_30d_pct_change DOUBLE,
    smoothed_total_indexes_30d_pct_change DOUBLE,
    smoothed_auditlog_30d_pct_change DOUBLE,
    smoothed_agg_total_calls_30d_pct_change DOUBLE,
    smoothed_agg_set_calls_30d_pct_change DOUBLE,
    smoothed_agg_match_calls_30d_pct_change DOUBLE,
    smoothed_agg_group_calls_30d_pct_change DOUBLE,
    smoothed_agg_project_calls_30d_pct_change DOUBLE,
    smoothed_agg_limit_calls_30d_pct_change DOUBLE,
    smoothed_agg_sort_calls_30d_pct_change DOUBLE,
    smoothed_agg_unwind_calls_30d_pct_change DOUBLE,
    smoothed_agg_add_fields_calls_30d_pct_change DOUBLE,
    smoothed_agg_lookup_calls_30d_pct_change DOUBLE,
    smoothed_agg_count_calls_30d_pct_change DOUBLE,
    smoothed_agg_skip_calls_30d_pct_change DOUBLE,
    smoothed_cluster_analytics_node_count_30d_pct_change DOUBLE,
    smoothed_adminapi_directapi_usage_30d_pct_change DOUBLE,
    smoothed_agg_change_stream_calls_30d_pct_change DOUBLE,
    smoothed_changestream_events_30d_pct_change DOUBLE,
    smoothed_total_hosts_transactions_transactionscollectionwritecount_30d_pct_change DOUBLE,
    smoothed_oa_1d_usage_filesize_30d_pct_change DOUBLE,
    smoothed_text_indexes_30d_pct_change DOUBLE,
    smoothed_text_active_1d_usage_30d_pct_change DOUBLE,
    smoothed_ts_collection_count_30d_pct_change DOUBLE,
    smoothed_trigger_events_30d_pct_change DOUBLE,
    smoothed_vector_indexes_30d_pct_change DOUBLE,
    smoothed_vector_active_1d_usage_30d_pct_change DOUBLE,
    smoothed_cluster_mrr_percentile_by_instance DOUBLE,
    smoothed_reads_writes_per_second_avg_percentile_by_instance DOUBLE,
    smoothed_data_size_total_gb_percentile_by_instance DOUBLE,
    smoothed_documents_total_percentile_by_instance DOUBLE,
    smoothed_database_count_percentile_by_instance DOUBLE,
    smoothed_collection_count_percentile_by_instance DOUBLE,
    smoothed_total_accesses_percentile_by_instance DOUBLE,
    smoothed_total_indexes_percentile_by_instance DOUBLE,
    smoothed_auditlog_percentile_by_instance DOUBLE,
    smoothed_agg_total_calls_percentile_by_instance DOUBLE,
    smoothed_agg_set_calls_percentile_by_instance DOUBLE,
    smoothed_agg_match_calls_percentile_by_instance DOUBLE,
    smoothed_agg_group_calls_percentile_by_instance DOUBLE,
    smoothed_agg_project_calls_percentile_by_instance DOUBLE,
    smoothed_agg_limit_calls_percentile_by_instance DOUBLE,
    smoothed_agg_sort_calls_percentile_by_instance DOUBLE,
    smoothed_agg_unwind_calls_percentile_by_instance DOUBLE,
    smoothed_agg_add_fields_calls_percentile_by_instance DOUBLE,
    smoothed_agg_lookup_calls_percentile_by_instance DOUBLE,
    smoothed_agg_count_calls_percentile_by_instance DOUBLE,
    smoothed_agg_skip_calls_percentile_by_instance DOUBLE,
    smoothed_cluster_analytics_node_count_percentile_by_instance DOUBLE,
    smoothed_adminapi_directapi_usage_percentile_by_instance DOUBLE,
    smoothed_agg_change_stream_calls_percentile_by_instance DOUBLE,
    smoothed_changestream_events_percentile_by_instance DOUBLE,
    smoothed_total_hosts_transactions_transactionscollectionwritecount_percentile_by_instance DOUBLE,
    smoothed_oa_1d_usage_filesize_percentile_by_instance DOUBLE,
    smoothed_text_indexes_percentile_by_instance DOUBLE,
    smoothed_text_active_1d_usage_percentile_by_instance DOUBLE,
    smoothed_ts_collection_count_percentile_by_instance DOUBLE,
    smoothed_trigger_events_percentile_by_instance DOUBLE,
    smoothed_vector_indexes_percentile_by_instance DOUBLE,
    smoothed_vector_active_1d_usage_percentile_by_instance DOUBLE,
    smoothed_cluster_mrr_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_reads_writes_per_second_avg_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_data_size_total_gb_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_documents_total_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_database_count_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_collection_count_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_total_accesses_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_total_indexes_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_auditlog_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_agg_total_calls_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_agg_set_calls_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_agg_match_calls_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_agg_group_calls_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_agg_project_calls_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_agg_limit_calls_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_agg_sort_calls_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_agg_unwind_calls_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_agg_add_fields_calls_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_agg_lookup_calls_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_agg_count_calls_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_agg_skip_calls_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_cluster_analytics_node_count_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_adminapi_directapi_usage_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_agg_change_stream_calls_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_changestream_events_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_total_hosts_transactions_transactionscollectionwritecount_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_oa_1d_usage_filesize_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_text_indexes_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_text_active_1d_usage_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_ts_collection_count_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_trigger_events_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_vector_indexes_30d_pct_change_percentile_by_instance DOUBLE,
    smoothed_vector_active_1d_usage_30d_pct_change_percentile_by_instance DOUBLE,
    core_product_engagement_score DOUBLE,
    core_product_trend_score DOUBLE,
    additional_feature_engagement_score DOUBLE,
    additional_feature_trend_score DOUBLE,
    demo_score DOUBLE,
    cluster_growth_score DOUBLE,
    demo_score_decile INTEGER,
    core_product_engagement_score_decile INTEGER,
    core_product_trend_score_decile INTEGER,
    additional_feature_engagement_score_decile INTEGER,
    additional_feature_trend_score_decile INTEGER,
    cluster_growth_decile INTEGER,
    weighted_score DOUBLE,
    processed_date DATE,
    ds DATE
)
WITH (
    format = 'PARQUET',
    partitioned_by = ARRAY['ds']
)
""")

conn.commit()
cursor.close()
conn.close()

print("Successfully created cluster_score_enterprise_flywheel table")

# %%
# Insert data in batches
BATCH_SIZE = 100

# Add processed_date column with today's date and convert scores to deciles
df_filtered["processed_date"] = pd.Timestamp.today().date()

# Convert timestamp columns to date strings in YYYY-MM-DD format
df_filtered['processed_date'] = pd.to_datetime(df_filtered['processed_date']).dt.strftime('%Y-%m-%d')
df_filtered['ds'] = pd.to_datetime(df_filtered['ds']).dt.strftime('%Y-%m-%d')

# Filter to only include accounts from landing_zone_list
valid_accounts = landing_zone_list['account_id'].unique()
filtered_scores = df_filtered[df_filtered['ultimate_parent_account_id'].isin(valid_accounts)]

# Round specific columns to 2 decimals
columns_to_round = [col for col in filtered_scores.columns if col.endswith('by_instance') or col.endswith('score')]
filtered_scores[columns_to_round] = filtered_scores[columns_to_round].round(2)

# Replace infinite values with NULL
filtered_scores = filtered_scores.replace([np.inf, -np.inf], None)

# Calculate number of batches
num_batches = len(filtered_scores) // BATCH_SIZE + (1 if len(filtered_scores) % BATCH_SIZE else 0)

total_inserted = 0

for batch in range(num_batches):
    start_idx = batch * BATCH_SIZE
    end_idx = min((batch + 1) * BATCH_SIZE, len(filtered_scores))
    
    batch_df = filtered_scores.iloc[start_idx:end_idx]
    
    conn = trino.dbapi.connect(
        host='presto-gateway.corp.mongodb.com',
        port=443,
        user='jiawei.zhou@mongodb.com', 
        catalog='awsdatacatalog',
        http_scheme='https',
        auth=trino.auth.BasicAuthentication("jiawei.zhou@mongodb.com", password),
        request_timeout=600  # Increase the timeout to 600 seconds
    )
    
    cursor = conn.cursor()
    
    # Build batch insert query
    values_list = []
    for _, row in batch_df.iterrows():
        # Replace NaN values with NULL
        row = row.where(pd.notnull(row), 'NULL')
        values_list.append(f"""(
            '{row['cluster_id']}',
            '{row['group_id']}',
            '{row['org_id']}',
            '{row['account_id']}',
            '{row['ultimate_parent_account_id']}',
            {row['cluster_age_month']},
            {row['cluster_analytics_node_count']},
            {row['cluster_mdb_major_version']},
            '{row['instance_size']}',
            {row['is_auto_expand_storage']},
            {row['is_auto_scaling_compute_enabled']},
            {row['is_auto_scaling_compute_scaledown_enabled']},
            {row['is_backup_selected']},
            {row['is_sharding']},
            {row['region_count']},
            {row['shard_count']},
            {row['total_accesses']},
            {row['total_indexes']},
            {row['agg_add_fields_calls']},
            {row['agg_change_stream_calls']},
            {row['agg_count_calls']},
            {row['agg_group_calls']},
            {row['agg_limit_calls']},
            {row['agg_lookup_calls']},
            {row['agg_match_calls']},
            {row['agg_project_calls']},
            {row['agg_set_calls']},
            {row['agg_skip_calls']},
            {row['agg_sort_calls']},
            {row['agg_unwind_calls']},
            {row['agg_total_calls']},
            {row['changestream_events']},
            {row['data_size_total_gb']},
            {row['documents_total']},
            {row['reads_writes_per_second_avg']},
            {row['collection_count']},
            {row['database_count']},
            {row['cluster_mrr']},
            {row['oa_1d_usage_filesize']},
            {row['trigger_events']},
            {row['text_active_1d_usage']},
            {row['text_indexes']},
            {row['vector_active_1d_usage']},
            {row['vector_indexes']},
            {row['ts_collection_count']},
            {row['total_hosts_transactions_transactionscollectionwritecount']},
            '{row['auditlog']}',
            {row['adminapi_directapi_usage']},
            {row['instance_size_numeric']},
            {row['smoothed_cluster_mrr']},
            {row['smoothed_reads_writes_per_second_avg']},
            {row['smoothed_data_size_total_gb']},
            {row['smoothed_documents_total']},
            {row['smoothed_database_count']},
            {row['smoothed_collection_count']},
            {row['smoothed_total_accesses']},
            {row['smoothed_total_indexes']},
            {row['smoothed_auditlog']},
            {row['smoothed_agg_total_calls']},
            {row['smoothed_agg_set_calls']},
            {row['smoothed_agg_match_calls']},
            {row['smoothed_agg_group_calls']},
            {row['smoothed_agg_project_calls']},
            {row['smoothed_agg_limit_calls']},
            {row['smoothed_agg_sort_calls']},
            {row['smoothed_agg_unwind_calls']},
            {row['smoothed_agg_add_fields_calls']},
            {row['smoothed_agg_lookup_calls']},
            {row['smoothed_agg_count_calls']},
            {row['smoothed_agg_skip_calls']},
            {row['smoothed_cluster_analytics_node_count']},
            {row['smoothed_adminapi_directapi_usage']},
            {row['smoothed_agg_change_stream_calls']},
            {row['smoothed_changestream_events']},
            {row['smoothed_total_hosts_transactions_transactionscollectionwritecount']},
            {row['smoothed_oa_1d_usage_filesize']},
            {row['smoothed_text_indexes']},
            {row['smoothed_text_active_1d_usage']},
            {row['smoothed_ts_collection_count']},
            {row['smoothed_trigger_events']},
            {row['smoothed_vector_indexes']},
            {row['smoothed_vector_active_1d_usage']},
            {row['smoothed_cluster_mrr_30d_pct_change']},
            {row['smoothed_reads_writes_per_second_avg_30d_pct_change']},
            {row['smoothed_data_size_total_gb_30d_pct_change']},
            {row['smoothed_documents_total_30d_pct_change']},
            {row['smoothed_database_count_30d_pct_change']},
            {row['smoothed_collection_count_30d_pct_change']},
            {row['smoothed_total_accesses_30d_pct_change']},
            {row['smoothed_total_indexes_30d_pct_change']},
            {row['smoothed_auditlog_30d_pct_change']},
            {row['smoothed_agg_total_calls_30d_pct_change']},
            {row['smoothed_agg_set_calls_30d_pct_change']},
            {row['smoothed_agg_match_calls_30d_pct_change']},
            {row['smoothed_agg_group_calls_30d_pct_change']},
            {row['smoothed_agg_project_calls_30d_pct_change']},
            {row['smoothed_agg_limit_calls_30d_pct_change']},
            {row['smoothed_agg_sort_calls_30d_pct_change']},
            {row['smoothed_agg_unwind_calls_30d_pct_change']},
            {row['smoothed_agg_add_fields_calls_30d_pct_change']},
            {row['smoothed_agg_lookup_calls_30d_pct_change']},
            {row['smoothed_agg_count_calls_30d_pct_change']},
            {row['smoothed_agg_skip_calls_30d_pct_change']},
            {row['smoothed_cluster_analytics_node_count_30d_pct_change']},
            {row['smoothed_adminapi_directapi_usage_30d_pct_change']},
            {row['smoothed_agg_change_stream_calls_30d_pct_change']},
            {row['smoothed_changestream_events_30d_pct_change']},
            {row['smoothed_total_hosts_transactions_transactionscollectionwritecount_30d_pct_change']},
            {row['smoothed_oa_1d_usage_filesize_30d_pct_change']},
            {row['smoothed_text_indexes_30d_pct_change']},
            {row['smoothed_text_active_1d_usage_30d_pct_change']},
            {row['smoothed_ts_collection_count_30d_pct_change']},
            {row['smoothed_trigger_events_30d_pct_change']},
            {row['smoothed_vector_indexes_30d_pct_change']},
            {row['smoothed_vector_active_1d_usage_30d_pct_change']},
            {row['smoothed_cluster_mrr_percentile_by_instance']},
            {row['smoothed_reads_writes_per_second_avg_percentile_by_instance']},
            {row['smoothed_data_size_total_gb_percentile_by_instance']},
            {row['smoothed_documents_total_percentile_by_instance']},
            {row['smoothed_database_count_percentile_by_instance']},
            {row['smoothed_collection_count_percentile_by_instance']},
            {row['smoothed_total_accesses_percentile_by_instance']},
            {row['smoothed_total_indexes_percentile_by_instance']},
            {row['smoothed_auditlog_percentile_by_instance']},
            {row['smoothed_agg_total_calls_percentile_by_instance']},
            {row['smoothed_agg_set_calls_percentile_by_instance']},
            {row['smoothed_agg_match_calls_percentile_by_instance']},
            {row['smoothed_agg_group_calls_percentile_by_instance']},
            {row['smoothed_agg_project_calls_percentile_by_instance']},
            {row['smoothed_agg_limit_calls_percentile_by_instance']},
            {row['smoothed_agg_sort_calls_percentile_by_instance']},
            {row['smoothed_agg_unwind_calls_percentile_by_instance']},
            {row['smoothed_agg_add_fields_calls_percentile_by_instance']},
            {row['smoothed_agg_lookup_calls_percentile_by_instance']},
            {row['smoothed_agg_count_calls_percentile_by_instance']},
            {row['smoothed_agg_skip_calls_percentile_by_instance']},
            {row['smoothed_cluster_analytics_node_count_percentile_by_instance']},
            {row['smoothed_adminapi_directapi_usage_percentile_by_instance']},
            {row['smoothed_agg_change_stream_calls_percentile_by_instance']},
            {row['smoothed_changestream_events_percentile_by_instance']},
            {row['smoothed_total_hosts_transactions_transactionscollectionwritecount_percentile_by_instance']},
            {row['smoothed_oa_1d_usage_filesize_percentile_by_instance']},
            {row['smoothed_text_indexes_percentile_by_instance']},
            {row['smoothed_text_active_1d_usage_percentile_by_instance']},
            {row['smoothed_ts_collection_count_percentile_by_instance']},
            {row['smoothed_trigger_events_percentile_by_instance']},
            {row['smoothed_vector_indexes_percentile_by_instance']},
            {row['smoothed_vector_active_1d_usage_percentile_by_instance']},
            {row['smoothed_cluster_mrr_30d_pct_change_percentile_by_instance']},
            {row['smoothed_reads_writes_per_second_avg_30d_pct_change_percentile_by_instance']},
            {row['smoothed_data_size_total_gb_30d_pct_change_percentile_by_instance']},
            {row['smoothed_documents_total_30d_pct_change_percentile_by_instance']},
            {row['smoothed_database_count_30d_pct_change_percentile_by_instance']},
            {row['smoothed_collection_count_30d_pct_change_percentile_by_instance']},
            {row['smoothed_total_accesses_30d_pct_change_percentile_by_instance']},
            {row['smoothed_total_indexes_30d_pct_change_percentile_by_instance']},
            {row['smoothed_auditlog_30d_pct_change_percentile_by_instance']},
            {row['smoothed_agg_total_calls_30d_pct_change_percentile_by_instance']},
            {row['smoothed_agg_set_calls_30d_pct_change_percentile_by_instance']},
            {row['smoothed_agg_match_calls_30d_pct_change_percentile_by_instance']},
            {row['smoothed_agg_group_calls_30d_pct_change_percentile_by_instance']},
            {row['smoothed_agg_project_calls_30d_pct_change_percentile_by_instance']},
            {row['smoothed_agg_limit_calls_30d_pct_change_percentile_by_instance']},
            {row['smoothed_agg_sort_calls_30d_pct_change_percentile_by_instance']},
            {row['smoothed_agg_unwind_calls_30d_pct_change_percentile_by_instance']},
            {row['smoothed_agg_add_fields_calls_30d_pct_change_percentile_by_instance']},
            {row['smoothed_agg_lookup_calls_30d_pct_change_percentile_by_instance']},
            {row['smoothed_agg_count_calls_30d_pct_change_percentile_by_instance']},
            {row['smoothed_agg_skip_calls_30d_pct_change_percentile_by_instance']},
            {row['smoothed_cluster_analytics_node_count_30d_pct_change_percentile_by_instance']},
            {row['smoothed_adminapi_directapi_usage_30d_pct_change_percentile_by_instance']},
            {row['smoothed_agg_change_stream_calls_30d_pct_change_percentile_by_instance']},
            {row['smoothed_changestream_events_30d_pct_change_percentile_by_instance']},
            {row['smoothed_total_hosts_transactions_transactionscollectionwritecount_30d_pct_change_percentile_by_instance']},
            {row['smoothed_oa_1d_usage_filesize_30d_pct_change_percentile_by_instance']},
            {row['smoothed_text_indexes_30d_pct_change_percentile_by_instance']},
            {row['smoothed_text_active_1d_usage_30d_pct_change_percentile_by_instance']},
            {row['smoothed_ts_collection_count_30d_pct_change_percentile_by_instance']},
            {row['smoothed_trigger_events_30d_pct_change_percentile_by_instance']},
            {row['smoothed_vector_indexes_30d_pct_change_percentile_by_instance']},
            {row['smoothed_vector_active_1d_usage_30d_pct_change_percentile_by_instance']},
            {row['core_product_engagement_score']},
            {row['core_product_trend_score']},
            {row['additional_feature_engagement_score']},
            {row['additional_feature_trend_score']},
            {row['demo_score']},
            {row['cluster_growth_score']},
            {row['demo_score_decile']},
            {row['core_product_engagement_score_decile']},
            {row['core_product_trend_score_decile']},
            {row['additional_feature_engagement_score_decile']},
            {row['additional_feature_trend_score_decile']},
            {row['cluster_growth_decile']},
            {row['weighted_score']},
            DATE '{row['processed_date']}',
            DATE '{row['ds']}'
        )""")
    
    values_str = ",\n".join(values_list)
    
    insert_query = f"""
    INSERT INTO product_analytics_internal.cluster_score_enterprise_flywheel (
        cluster_id,
        group_id,
        org_id,
        account_id,
        ultimate_parent_account_id,
        cluster_age_month,
        cluster_analytics_node_count,
        cluster_mdb_major_version,
        instance_size,
        is_auto_expand_storage,
        is_auto_scaling_compute_enabled,
        is_auto_scaling_compute_scaledown_enabled,
        is_backup_selected,
        is_sharding,
        region_count,
        shard_count,
        total_accesses,
        total_indexes,
        agg_add_fields_calls,
        agg_change_stream_calls,
        agg_count_calls,
        agg_group_calls,
        agg_limit_calls,
        agg_lookup_calls,
        agg_match_calls,
        agg_project_calls,
        agg_set_calls,
        agg_skip_calls,
        agg_sort_calls,
        agg_unwind_calls,
        agg_total_calls,
        changestream_events,
        data_size_total_gb,
        documents_total,
        reads_writes_per_second_avg,
        collection_count,
        database_count,
        cluster_mrr,
        oa_1d_usage_filesize,
        trigger_events,
        text_active_1d_usage,
        text_indexes,
        vector_active_1d_usage,
        vector_indexes,
        ts_collection_count,
        total_hosts_transactions_transactionscollectionwritecount,
        auditlog,
        adminapi_directapi_usage,
        instance_size_numeric,
        smoothed_cluster_mrr,
        smoothed_reads_writes_per_second_avg,
        smoothed_data_size_total_gb,
        smoothed_documents_total,
        smoothed_database_count,
        smoothed_collection_count,
        smoothed_total_accesses,
        smoothed_total_indexes,
        smoothed_auditlog,
        smoothed_agg_total_calls,
        smoothed_agg_set_calls,
        smoothed_agg_match_calls,
        smoothed_agg_group_calls,
        smoothed_agg_project_calls,
        smoothed_agg_limit_calls,
        smoothed_agg_sort_calls,
        smoothed_agg_unwind_calls,
        smoothed_agg_add_fields_calls,
        smoothed_agg_lookup_calls,
        smoothed_agg_count_calls,
        smoothed_agg_skip_calls,
        smoothed_cluster_analytics_node_count,
        smoothed_adminapi_directapi_usage,
        smoothed_agg_change_stream_calls,
        smoothed_changestream_events,
        smoothed_total_hosts_transactions_transactionscollectionwritecount,
        smoothed_oa_1d_usage_filesize,
        smoothed_text_indexes,
        smoothed_text_active_1d_usage,
        smoothed_ts_collection_count,
        smoothed_trigger_events,
        smoothed_vector_indexes,
        smoothed_vector_active_1d_usage,
        smoothed_cluster_mrr_30d_pct_change,
        smoothed_reads_writes_per_second_avg_30d_pct_change,
        smoothed_data_size_total_gb_30d_pct_change,
        smoothed_documents_total_30d_pct_change,
        smoothed_database_count_30d_pct_change,
        smoothed_collection_count_30d_pct_change,
        smoothed_total_accesses_30d_pct_change,
        smoothed_total_indexes_30d_pct_change,
        smoothed_auditlog_30d_pct_change,
        smoothed_agg_total_calls_30d_pct_change,
        smoothed_agg_set_calls_30d_pct_change,
        smoothed_agg_match_calls_30d_pct_change,
        smoothed_agg_group_calls_30d_pct_change,
        smoothed_agg_project_calls_30d_pct_change,
        smoothed_agg_limit_calls_30d_pct_change,
        smoothed_agg_sort_calls_30d_pct_change,
        smoothed_agg_unwind_calls_30d_pct_change,
        smoothed_agg_add_fields_calls_30d_pct_change,
        smoothed_agg_lookup_calls_30d_pct_change,
        smoothed_agg_count_calls_30d_pct_change,
        smoothed_agg_skip_calls_30d_pct_change,
        smoothed_cluster_analytics_node_count_30d_pct_change,
        smoothed_adminapi_directapi_usage_30d_pct_change,
        smoothed_agg_change_stream_calls_30d_pct_change,
        smoothed_changestream_events_30d_pct_change,
        smoothed_total_hosts_transactions_transactionscollectionwritecount_30d_pct_change,
        smoothed_oa_1d_usage_filesize_30d_pct_change,
        smoothed_text_indexes_30d_pct_change,
        smoothed_text_active_1d_usage_30d_pct_change,
        smoothed_ts_collection_count_30d_pct_change,
        smoothed_trigger_events_30d_pct_change,
        smoothed_vector_indexes_30d_pct_change,
        smoothed_vector_active_1d_usage_30d_pct_change,
        smoothed_cluster_mrr_percentile_by_instance,
        smoothed_reads_writes_per_second_avg_percentile_by_instance,
        smoothed_data_size_total_gb_percentile_by_instance,
        smoothed_documents_total_percentile_by_instance,
        smoothed_database_count_percentile_by_instance,
        smoothed_collection_count_percentile_by_instance,
        smoothed_total_accesses_percentile_by_instance,
        smoothed_total_indexes_percentile_by_instance,
        smoothed_auditlog_percentile_by_instance,
        smoothed_agg_total_calls_percentile_by_instance,
        smoothed_agg_set_calls_percentile_by_instance,
        smoothed_agg_match_calls_percentile_by_instance,
        smoothed_agg_group_calls_percentile_by_instance,
        smoothed_agg_project_calls_percentile_by_instance,
        smoothed_agg_limit_calls_percentile_by_instance,
        smoothed_agg_sort_calls_percentile_by_instance,
        smoothed_agg_unwind_calls_percentile_by_instance,
        smoothed_agg_add_fields_calls_percentile_by_instance,
        smoothed_agg_lookup_calls_percentile_by_instance,
        smoothed_agg_count_calls_percentile_by_instance,
        smoothed_agg_skip_calls_percentile_by_instance,
        smoothed_cluster_analytics_node_count_percentile_by_instance,
        smoothed_adminapi_directapi_usage_percentile_by_instance,
        smoothed_agg_change_stream_calls_percentile_by_instance,
        smoothed_changestream_events_percentile_by_instance,
        smoothed_total_hosts_transactions_transactionscollectionwritecount_percentile_by_instance,
        smoothed_oa_1d_usage_filesize_percentile_by_instance,
        smoothed_text_indexes_percentile_by_instance,
        smoothed_text_active_1d_usage_percentile_by_instance,
        smoothed_ts_collection_count_percentile_by_instance,
        smoothed_trigger_events_percentile_by_instance,
        smoothed_vector_indexes_percentile_by_instance,
        smoothed_vector_active_1d_usage_percentile_by_instance,
        smoothed_cluster_mrr_30d_pct_change_percentile_by_instance,
        smoothed_reads_writes_per_second_avg_30d_pct_change_percentile_by_instance,
        smoothed_data_size_total_gb_30d_pct_change_percentile_by_instance,
        smoothed_documents_total_30d_pct_change_percentile_by_instance,
        smoothed_database_count_30d_pct_change_percentile_by_instance,
        smoothed_collection_count_30d_pct_change_percentile_by_instance,
        smoothed_total_accesses_30d_pct_change_percentile_by_instance,
        smoothed_total_indexes_30d_pct_change_percentile_by_instance,
        smoothed_auditlog_30d_pct_change_percentile_by_instance,
        smoothed_agg_total_calls_30d_pct_change_percentile_by_instance,
        smoothed_agg_set_calls_30d_pct_change_percentile_by_instance,
        smoothed_agg_match_calls_30d_pct_change_percentile_by_instance,
        smoothed_agg_group_calls_30d_pct_change_percentile_by_instance,
        smoothed_agg_project_calls_30d_pct_change_percentile_by_instance,
        smoothed_agg_limit_calls_30d_pct_change_percentile_by_instance,
        smoothed_agg_sort_calls_30d_pct_change_percentile_by_instance,
        smoothed_agg_unwind_calls_30d_pct_change_percentile_by_instance,
        smoothed_agg_add_fields_calls_30d_pct_change_percentile_by_instance,
        smoothed_agg_lookup_calls_30d_pct_change_percentile_by_instance,
        smoothed_agg_count_calls_30d_pct_change_percentile_by_instance,
        smoothed_agg_skip_calls_30d_pct_change_percentile_by_instance,
        smoothed_cluster_analytics_node_count_30d_pct_change_percentile_by_instance,
        smoothed_adminapi_directapi_usage_30d_pct_change_percentile_by_instance,
        smoothed_agg_change_stream_calls_30d_pct_change_percentile_by_instance,
        smoothed_changestream_events_30d_pct_change_percentile_by_instance,
        smoothed_total_hosts_transactions_transactionscollectionwritecount_30d_pct_change_percentile_by_instance,
        smoothed_oa_1d_usage_filesize_30d_pct_change_percentile_by_instance,
        smoothed_text_indexes_30d_pct_change_percentile_by_instance,
        smoothed_text_active_1d_usage_30d_pct_change_percentile_by_instance,
        smoothed_ts_collection_count_30d_pct_change_percentile_by_instance,
        smoothed_trigger_events_30d_pct_change_percentile_by_instance,
        smoothed_vector_indexes_30d_pct_change_percentile_by_instance,
        smoothed_vector_active_1d_usage_30d_pct_change_percentile_by_instance,
        core_product_engagement_score,
        core_product_trend_score,
        additional_feature_engagement_score,
        additional_feature_trend_score,
        demo_score,
        cluster_growth_score,
        demo_score_decile,
        core_product_engagement_score_decile,
        core_product_trend_score_decile,
        additional_feature_engagement_score_decile,
        additional_feature_trend_score_decile,
        cluster_growth_decile,
        weighted_score,
        processed_date,
        ds
    ) VALUES {values_str}
    """
    
    cursor.execute(insert_query)
    conn.commit()
    cursor.close()
    conn.close()
    
    total_inserted += len(batch_df)
    print(f"Inserted batch {batch + 1}/{num_batches} ({total_inserted}/{len(filtered_scores)} rows)")

print(f"Successfully inserted all {total_inserted} rows of data")


# %%
print(f"Target date: {df_filtered['ds'].iloc[0]}")



