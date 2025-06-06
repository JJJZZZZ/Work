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
import os
import keyring
import json
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import yaml 
import logging # Added logging import

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # This initial print is okay as logging is not yet configured
        print(f"Configuration loaded from {config_path}") 
        return config
    except FileNotFoundError:
        # This print is okay for the same reason
        print(f"CRITICAL: Configuration file not found at {config_path}")
        raise SystemExit(f"CRITICAL: Configuration file {config_path} not found.")
    except yaml.YAMLError as e:
        print(f"CRITICAL: Error parsing YAML configuration: {e}")
        raise SystemExit(f"CRITICAL: Error parsing YAML configuration: {e}")

def setup_logging(log_config, script_logger_name):
    """Configures basic logging based on config."""
    log_level_str = log_config.get('level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    log_formatter = logging.Formatter(log_config.get('format', "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    logger = logging.getLogger(script_logger_name)
    logger.setLevel(log_level) 
    
    # Prevent adding multiple handlers if this function is called again (though it shouldn't be)
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)
        
        # File Handler
        file_handler = logging.FileHandler(log_config.get('log_file', 'scoring_auto.log'))
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    
    logger.info("Logging configured: Level %s, File %s", log_level_str, log_config.get('log_file'))
    return logger

def setup_environment():
    """Configures pandas display options."""
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    # No logger needed in setup_environment itself as it's very simple.

def get_user_inputs(config):
    """Gets user credentials and model scoring date from input."""
    # Get the main script logger name from config, then create a child logger.
    script_logger_name = config['logging'].get('script_logger_name', 'scoring_script')
    logger = logging.getLogger(script_logger_name + '.get_user_inputs')
    logger.debug("Entering get_user_inputs function.")
    
    def get_credentials(app_config):
        cred_logger = logging.getLogger(script_logger_name + '.get_credentials') # Child logger
        cred_logger.info("Attempting to retrieve credentials.")
        username = None
        password = None
        
        keyring_service = app_config['other'].get('keyring_service_name', 'trino_db')
        username_for_keyring_lookup = app_config['database'].get('username', 'jiawei.zhou@mongodb.com')
        cred_logger.debug(f"Keyring service: {keyring_service}, Username for lookup: {username_for_keyring_lookup}")

        # 1. Try Keyring
        try:
            username = username_for_keyring_lookup 
            password = keyring.get_password(keyring_service, username)
            if password:
                cred_logger.info(f"Credentials for {username} retrieved from keyring.")
            else:
                username = None 
                cred_logger.debug("Password not found in keyring for the specified username.")
        except Exception as e:
            cred_logger.warning(f"Keyring access failed: {e}") # Changed from print to logger.warning
            username = None 
            password = None

        # 2. Try Environment Variables
        if not password: 
            env_username = os.environ.get('TRINO_USERNAME')
            env_password = os.environ.get('TRINO_PASSWORD')
            if env_username and env_password:
                cred_logger.info("Credentials retrieved from environment variables (TRINO_USERNAME, TRINO_PASSWORD).")
                username = env_username
                password = env_password
            elif env_username and not env_password: 
                 username = env_username 
                 cred_logger.info(f"Username '{username}' retrieved from TRINO_USERNAME. Password not found in TRINO_PASSWORD.")
            elif not env_username and env_password: 
                username = username_for_keyring_lookup 
                password = env_password
                cred_logger.info(f"Password retrieved from TRINO_PASSWORD. Using username '{username}' from config/default.")
            else:
                cred_logger.debug("TRINO_USERNAME or TRINO_PASSWORD not found in environment variables.")

        # 3. Configuration File for Username
        if not username: 
            username = app_config['database'].get('username')
            if username:
                cred_logger.info(f"Username '{username}' retrieved from config.yaml.")
        
        if not username: 
            username = "jiawei.zhou@mongodb.com" 
            cred_logger.info(f"Username defaulted to '{username}'.")

        # 4. Backup JSON config file for password
        if not password:
            backup_config_filename = app_config['other'].get('trino_backup_config_path', '.trino_config.json')
            backup_config_path = Path.home() / backup_config_filename
            cred_logger.debug(f"Checking backup JSON config for password: {backup_config_path}")
            if backup_config_path.exists():
                try:
                    with open(backup_config_path, "r") as f:
                        backup_cfg = json.load(f)
                        if backup_cfg.get("username") == username: 
                            password = backup_cfg.get("password")
                            if password:
                                cred_logger.info(f"Password for {username} retrieved from backup JSON config: {backup_config_path}")
                except json.JSONDecodeError:
                    cred_logger.warning(f"Could not decode JSON from {backup_config_path}") # Was print
                except Exception as e:
                    cred_logger.error(f"An error occurred reading {backup_config_path}: {e}", exc_info=True) # Was print

        # 5. getpass.getpass() as final fallback for password
        if not password:
            # This print is kept as it is a direct user prompt when other methods fail.
            print(f"Password for {username} not found through other methods. Please enter it manually.")
            password = getpass.getpass(f"Enter password for {username}: ") # Remains getpass
            
            save_choice = input("Save credentials for future use? (keyring/backup_json/both/none) [none]: ").lower() # Remains input
            if save_choice in ["keyring", "both"]:
                try:
                    keyring.set_password(keyring_service, username, password)
                    cred_logger.info("Password saved to keyring.") # Was print
                except Exception as e:
                    cred_logger.error(f"Failed to save password to keyring: {e}", exc_info=True) # Was print
            if save_choice in ["backup_json", "both"]:
                backup_config_filename = app_config['other'].get('trino_backup_config_path', '.trino_config.json')
                backup_config_path = Path.home() / backup_config_filename
                try:
                    with open(backup_config_path, "w") as f:
                        json.dump({"username": username, "password": password}, f)
                    cred_logger.info(f"Credentials saved to {backup_config_path}.") # Was print
                except Exception as e:
                    cred_logger.error(f"Error saving credentials to {backup_config_path}: {e}", exc_info=True) # Was print
        
        return username, password

    username, password = get_credentials(config) # Pass full config to get_credentials
    logger.info(f"Credentials obtained for username: {username}") # Log after get_credentials call
    
    model_score_date_str = input("Enter the date for model scoring (YYYY-MM-DD): ") # This remains an input
    end_date_obj = datetime.strptime(model_score_date_str, '%Y-%m-%d')
    start_date_obj = end_date_obj - timedelta(days=30)
    start_date_str = start_date_obj.strftime('%Y-%m-%d')
    logger.info(f"Date range for scoring: {start_date_str} to {model_score_date_str}")
    
    return username, password, start_date_str, model_score_date_str, end_date_obj

def connect_db(username, password, db_config, script_logger_name): # Added script_logger_name
    """Establishes a connection to the Trino database using config."""
    logger = logging.getLogger(script_logger_name + '.connect_db')
    logger.info(f"Attempting to connect to Trino host: {db_config['host']} as user: {username}")
    try:
        conn = trino.dbapi.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=username, 
            catalog=db_config['catalog'],
            http_scheme=db_config['http_scheme'],
            auth=trino.auth.BasicAuthentication(username, password),
            request_timeout=db_config.get('request_timeout', 600) 
        )
        logger.info(f"Successfully connected to Trino host: {db_config['host']}")
        return conn
    except Exception as e:
        logger.error(f"Trino connection failed for host {db_config['host']}: {e}", exc_info=True)
        raise # Re-raise the exception to be handled by the caller

def fetch_data(conn, start_date, model_score_date, query_template, script_logger_name): # Added script_logger_name
    """Fetches data from the database for the given date range using a query template."""
    logger = logging.getLogger(script_logger_name + '.fetch_data')
    logger.info(f"Fetching data from {start_date} to {model_score_date}")
    cur = None # Ensure cursor is defined for finally block
    try:
        cur = conn.cursor()
        query = query_template.format(start_date=start_date, model_score_date=model_score_date)
        logger.debug(f"Executing query (first 200 chars): {query[:200]}...")
        cur.execute(query)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=columns)
        logger.info(f"Fetched {len(df)} rows and {len(columns)} columns.")
    except Exception as e:
        logger.error(f"Failed to execute query or fetch data: {e}", exc_info=True)
        raise 
    finally:
        if cur:
            cur.close()
    return df

def preprocess_data(df_backup, script_logger_name): # Added script_logger_name
    """Performs initial data preprocessing."""
    logger = logging.getLogger(script_logger_name + '.preprocess_data')
    logger.info(f"Starting preprocessing on DataFrame with {len(df_backup)} rows.")
    df = df_backup.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    
    date_counts = df.groupby('ds').size()
    logger.info("Number of rows per date:\n%s", date_counts.to_string())
    logger.info(f"Total number of unique dates: {len(date_counts)}")

    df["instance_size_numeric"] = df["instance_size"].str.extract(r"(\d+)").astype(float)
    df["cluster_mdb_major_version"] = pd.to_numeric(
        df["cluster_mdb_major_version"], errors="coerce"
    )
    logger.debug("Converted 'instance_size' and 'cluster_mdb_major_version' to numeric.")
    logger.debug("Head of relevant columns after preprocessing:\n%s", 
                 df[["instance_size", "instance_size_numeric", "cluster_mdb_major_version"]].head().to_string())
    logger.info("Preprocessing complete.")
    return df

# --- Main script execution starts here ---
# setup_environment() # Call this if you want the pandas options set globally

# username, password, start_date, model_score_date, end_date_obj = get_user_inputs()
# conn = connect_db(username, password)
# df_backup = fetch_data(conn, start_date, model_score_date)
# conn.close() # Close connection after fetching data

# df = preprocess_data(df_backup)
# df.head() # Displaying the first few rows of the DataFrame


def calculate_rolling_metrics(df, columns_to_smooth_list):
    """
    Calculates rolling averages and percentage changes for key metrics.
    Uses columns_to_smooth_list from config.
    """
    # Sort chronologically by cluster before calculations
def calculate_rolling_metrics(df, columns_to_smooth_list, script_logger_name): # Added script_logger_name
    """
    Calculates rolling averages and percentage changes for key metrics.
    Uses columns_to_smooth_list from config.
    """
    logger = logging.getLogger(script_logger_name + '.calculate_rolling_metrics')
    logger.info("Calculating rolling metrics...")
    df = df.sort_values(["cluster_id", "ds"])

    for col in columns_to_smooth_list: 
        smoothed_col = f"smoothed_{col}"
        df[smoothed_col] = df.groupby("cluster_id")[col].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        logger.debug(f"Calculated smoothed column: {smoothed_col}")

    for col in columns_to_smooth_list: 
        smoothed_col = f"smoothed_{col}"
        df[f"{smoothed_col}_30d_pct_change"] = df.groupby("cluster_id")[
            smoothed_col
        ].transform(lambda x: x.pct_change(periods=30, fill_method=None))
        logger.debug(f"Calculated 30d_pct_change for: {smoothed_col}")
    logger.info("Rolling metrics calculation complete.")
    return df

def calculate_percentile_ranks(df, end_date_obj, script_logger_name): # Added script_logger_name
    """
    Calculates relative usage percentiles for smoothed metrics.
    Filters data for the end_date_obj.
    """
    logger = logging.getLogger(script_logger_name + '.calculate_percentile_ranks')
    logger.info(f"Calculating percentile ranks for date: {end_date_obj.strftime('%Y-%m-%d')}")
    df_filtered = df[df['ds'] == end_date_obj].copy() 
    
    if df_filtered.empty:
        logger.warning(f"No data found for date {end_date_obj.strftime('%Y-%m-%d')} when calculating percentile ranks.")
        return df_filtered

    smoothed_columns = [col for col in df_filtered.columns if col.startswith("smoothed")] 
    df_filtered[smoothed_columns] = df_filtered[smoothed_columns].fillna(0)
    logger.debug(f"Found {len(smoothed_columns)} smoothed columns for percentile calculation.")

    def percentile_rank_func(series):
        return series.rank(pct=True)

    percentile_ranks_data = {}
    for col in smoothed_columns:
        percentile_ranks_data[f"{col}_percentile_by_instance"] = df_filtered.groupby("instance_size")[col].transform(percentile_rank_func)
        logger.debug(f"Calculated percentile rank for: {col}")

    df_filtered = pd.concat([df_filtered, pd.DataFrame(percentile_ranks_data)], axis=1)
    
    logger.info("Percentile rank calculation complete.")
    logger.debug("Head of df_filtered after percentile rank calculation:\n%s", df_filtered.head().to_string()) # Was print
    return df_filtered

# df = calculate_rolling_metrics(df, config['feature_engineering']['columns_to_smooth'], config['logging']['script_logger_name'])
# df_filtered = calculate_percentile_ranks(df, end_date_obj, config['logging']['script_logger_name'])

def apply_model(df, features_list, model_base_path, model_filename, score_column_name, script_logger_name): # Added script_logger_name
    """Loads a pre-trained model and applies it to the dataframe to generate scores."""
    logger = logging.getLogger(script_logger_name + '.apply_model')
    model_full_path = os.path.join(model_base_path, model_filename)
    logger.info(f"Applying model {model_filename} for score '{score_column_name}'. Features count: {len(features_list)}")
    logger.debug(f"Model path: {model_full_path}, Features: {features_list}")
    try:
        pipeline = joblib.load(model_full_path)
        df[score_column_name] = pipeline.predict_proba(df[features_list])[:, 1]
        logger.info(f"Successfully applied model {model_filename} to create column {score_column_name}.")
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_full_path}")
        df[score_column_name] = np.nan 
    except Exception as e:
        logger.error(f"An error occurred while applying model {model_full_path}: {e}", exc_info=True)
        df[score_column_name] = np.nan
    return df

# Feature lists will be loaded from config in main() and passed to apply_model calls
# Example:
# features_core_engagement = config['model_features']['core_engagement']
# df_filtered = apply_model(df_filtered, features_core_engagement, 
#                           config['paths']['model_directory'], 
#                           config['model_files']['core_product_engagement'], 
#                           'core_product_engagement_score')
# ... similar calls for other models

def calculate_deciles(df, score_columns_map, script_logger_name): # Added script_logger_name
    """Calculates deciles (1-10) for specified score columns."""
    logger = logging.getLogger(script_logger_name + '.calculate_deciles')
    logger.info(f"Calculating deciles for columns: {list(score_columns_map.keys())}")
    for original_col, decile_col_name in score_columns_map.items():
        if original_col in df:
            if df[original_col].notna().sum() > 0 and df[original_col].nunique() > 1: # Check for NaNs and unique values
                 try:
                    df[decile_col_name] = pd.qcut(df[original_col].rank(method='first'), q=10, labels=False, duplicates='drop') + 1
                    logger.debug(f"Calculated decile for {original_col} into {decile_col_name}.")
                 except Exception as e:
                    logger.error(f"Error calculating decile for {original_col}: {e}. Assigning NaN.", exc_info=True)
                    df[decile_col_name] = np.nan 
            else:
                logger.warning(f"Column {original_col} has all NaNs or too few unique values for decile calculation. Assigning NaN to {decile_col_name}.")
                df[decile_col_name] = np.nan
        else:
            logger.warning(f"Column {original_col} not found in DataFrame for decile calculation. Assigning NaN to {decile_col_name}.")
            df[decile_col_name] = np.nan # Ensure column exists
    logger.info("Decile calculation complete.")
    return df

# score_columns_to_calculate_deciles = {
# "demo_score": "demo_score_decile",
# "core_product_engagement_score": "core_product_engagement_score_decile",
# "core_product_trend_score": "core_product_trend_score_decile",
# "additional_feature_engagement_score": "additional_feature_engagement_score_decile",
# "additional_feature_trend_score": "additional_feature_trend_score_decile",
# "cluster_growth_score": "cluster_growth_decile",
# }
# df_filtered = calculate_deciles(df_filtered, score_columns_to_calculate_deciles)

# Plot growth score distributions (can be moved to a main/reporting function)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
# sns.histplot(df_filtered["cluster_growth_score"], ax=ax1)
# ax1.set_title("Distribution of Cluster Growth Score")
# ax1.set_xlabel("Cluster Growth Score")
# ax1.set_ylabel("Count")
# sns.histplot(df_filtered["cluster_growth_decile"], ax=ax2)
# ax2.set_title("Distribution of Cluster Growth Score (1-10 Scale)")
# ax2.set_xlabel("Cluster Growth Score (1-10)")
# ax2.set_ylabel("Count")
# plt.tight_layout()
# plt.show()

# print("df_filtered head after decile calculation:") # Optional
# df_filtered.head() # Optional

def aggregate_account_scores(df_filtered, group_by_col, mrr_col, growth_score_col, ds_col, script_logger_name): # Added script_logger_name
    """Aggregates cluster scores to the account level."""
    logger = logging.getLogger(script_logger_name + '.aggregate_account_scores')
    logger.info(f"Aggregating scores to account level by '{group_by_col}'.")
    
    df_filtered_copy = df_filtered.copy() 
    df_filtered_copy["weighted_score"] = df_filtered_copy[growth_score_col] * df_filtered_copy[mrr_col]
    
    agg_dict = {
        growth_score_col: "sum", 
        mrr_col: "sum",          
        ds_col: "max"            
    }
    logger.debug(f"Aggregation dictionary: {agg_dict}")
    
    account_scores_df = df_filtered_copy.groupby(group_by_col).agg(
        weighted_score_sum=(growth_score_col, "sum"),
        mrr_sum=(mrr_col, "sum"),
        ds_max=(ds_col, "max")
    ).reset_index()

    account_scores_df["account_growth_score"] = account_scores_df["weighted_score_sum"] / account_scores_df["mrr_sum"]
    
    account_score_decile_map = {"account_growth_score": "account_growth_decile"}
    account_scores_df = calculate_deciles(account_scores_df, account_score_decile_map, script_logger_name) 

    account_scores_df["processed_date"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    
    account_scores_df.rename(columns={
        "weighted_score_sum": "weighted_score", 
        "mrr_sum": "smoothed_cluster_mrr",  
        "ds_max": "ds" 
    }, inplace=True)

    logger.info(f"Account aggregation complete. {len(account_scores_df)} accounts processed.")
    logger.debug("Head of account_scores_df:\n%s", account_scores_df.head().to_string())
    return account_scores_df

# account_scores_df = aggregate_account_scores(df_filtered, "ultimate_parent_account_id", 
#                                            "smoothed_cluster_mrr", "cluster_growth_score", "ds", config['logging']['script_logger_name'])

# Plotting for account_scores (can be moved to a main/reporting function)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
# sns.histplot(account_scores["account_growth_score"], ax=ax1)
# ax1.set_title("Distribution of Account Growth Score")
# ax1.set_xlabel("Account Growth Score") 
# ax1.set_ylabel("Count")
# sns.histplot(account_scores["account_growth_decile"], ax=ax2)
# ax2.set_title("Distribution of Account Growth Score (1-10 Scale)")
# ax2.set_xlabel("Account Growth Score (1-10)")
# ax2.set_ylabel("Count")
# plt.tight_layout()
# plt.show()

def load_landing_zone_list(file_path, use_cols, column_mapping, script_logger_name): # Added script_logger_name
    """Loads the landing zone list CSV."""
    logger = logging.getLogger(script_logger_name + '.load_landing_zone_list')
    logger.info(f"Loading landing zone list from: {file_path}")
    try:
        df = pd.read_csv(file_path, usecols=use_cols)
        df.columns = df.columns.str.strip().str.lower()
        df = df.rename(columns=column_mapping)
        logger.info(f"Landing zone list loaded successfully. {len(df)} rows found.")
        logger.debug("Head of landing_zone_list:\n%s", df.head().to_string())
        return df
    except FileNotFoundError:
        logger.error(f"Landing zone file not found at {file_path}")
        return pd.DataFrame() 
    except Exception as e:
        logger.error(f"An error occurred while loading landing zone list: {e}", exc_info=True)
        return pd.DataFrame()

# landing_zone_list_df = load_landing_zone_list(
#       config['paths']['landing_zone_csv'],
#       config['landing_zone_config']['use_cols'],
#       config['landing_zone_config']['column_mapping'],
#       config['logging']['script_logger_name']
# )
# landing_zone_columns = ['Account Name', 'Account ID']
# landing_zone_map = {'account name': 'account_name', 'account id': 'account_id'}
# landing_zone_list = load_landing_zone_list(landing_zone_file, landing_zone_columns, landing_zone_map)

# --- Database Table Creation and Data Insertion ---

# Constants for DB connection details will be loaded from config.
# ACCOUNT_TABLE_NAME and CLUSTER_TABLE_NAME will be loaded from config.

def get_db_connection(username, password, db_config): # db_config from main config file
    """Creates and returns a Trino database connection using config."""
    return trino.dbapi.connect(
        host=db_config['host'],
        port=db_config['port'],
        user=username,
        catalog=db_config['catalog'],
        http_scheme=db_config['http_scheme'],
        auth=trino.auth.BasicAuthentication(username, password),
        request_timeout=db_config.get('request_timeout', 600) # Use get for optional param
    )

def create_table(username, password, db_config, table_sql_template, table_name_from_config_key):
    """Creates a table in the database if it doesn't already exist."""
    # table_name is now retrieved from db_config using table_name_from_config_key
    table_full_name = db_config[table_name_from_config_key]
    # The SQL template should now be formatted with the actual table name
    # Assuming table_sql_template might need formatting if it were generic.
    # For now, specific SQLs are still defined globally but will use config for table names.
    
    # Reconstruct SQL with table name from config for ACCOUNT_TABLE_SQL and CLUSTER_TABLE_SQL
    # This part needs to be handled carefully if SQL templates are in config.
    # For now, the SQLs are global but will be modified to use config table names.
    
    # The existing global ACCOUNT_TABLE_SQL and CLUSTER_TABLE_SQL will be modified
    # to use f-strings with table names from the config object passed into main.
    # So, table_sql_template argument will be the already formatted SQL string.

    try:
        with get_db_connection(username, password, db_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute(table_sql_template) # table_sql_template is now the fully formatted SQL
            conn.commit()
        print(f"Successfully ensured table {table_full_name} exists.")
    except Exception as e:
        print(f"Error creating table {table_full_name}: {e}")

# SQL definitions will be constructed in main() using config values for table names.
# Example:
# ACCOUNT_TABLE_SQL = f"""CREATE TABLE IF NOT EXISTS {config['database']['account_score_table']} ( ... )"""
# CLUSTER_TABLE_SQL = f"""CREATE TABLE IF NOT EXISTS {config['database']['cluster_score_table']} ( ... )"""


def insert_account_scores_batchwise(username, password, db_config, df_scores, landing_zone_df, batch_size):
    """Inserts account scores into the database in batches."""
    account_table_name = db_config['account_score_table']
    if df_scores.empty:
        print("Account scores DataFrame is empty. Nothing to insert.")
        return
    if landing_zone_df.empty:
        print("Landing zone DataFrame is empty. Cannot filter accounts for insertion.")
        return

    df_scores['ds'] = pd.to_datetime(df_scores['ds']).dt.date
    df_scores['processed_date'] = pd.to_datetime(df_scores['processed_date']).dt.date

    valid_accounts = landing_zone_df['account_id'].unique()
    df_to_insert = df_scores[df_scores['ultimate_parent_account_id'].isin(valid_accounts)].copy() 
    
    if df_to_insert.empty:
        print("No account scores to insert after filtering by landing zone list.")
        return

    num_batches = len(df_to_insert) // batch_size + (1 if len(df_to_insert) % batch_size else 0)
    total_inserted = 0

    try:
        with get_db_connection(username, password, db_config) as conn:
            for batch_num in range(num_batches):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, len(df_to_insert))
                batch_df = df_to_insert.iloc[start_idx:end_idx]

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
                INSERT INTO {account_table_name} (
                    ultimate_parent_account_id, weighted_score, smoothed_cluster_mrr,
                    account_growth_score, account_growth_decile, processed_date, ds
                ) VALUES {values_str}"""
                
                with conn.cursor() as cursor:
                    cursor.execute(insert_query)
                conn.commit() 
                
                total_inserted += len(batch_df)
                print(f"Account Scores: Inserted batch {batch_num + 1}/{num_batches} ({len(batch_df)} rows). Total inserted: {total_inserted}")
        print(f"Successfully inserted all {total_inserted} rows of account score data.")
    except Exception as e:
        print(f"Error inserting account scores: {e}")

# insert_account_scores_batchwise(username, password, account_scores, landing_zone_list)

# ... (missing accounts check remains the same, just ensure correct DataFrames are passed)


def insert_cluster_scores_batchwise(username, password, db_config, df_scores, landing_zone_df, batch_size):
    """Inserts cluster scores into the database in batches."""
    cluster_table_name = db_config['cluster_score_table']
    if df_scores.empty:
        print("Cluster scores DataFrame is empty. Nothing to insert.")
        return
    if landing_zone_df.empty:
        print("Landing zone DataFrame is empty. Cannot filter accounts for insertion.")
        return

# Global constants for table SQL definitions will be removed and constructed in main() using config.

# def insert_account_scores_batchwise(username, password, db_config, df_scores, landing_zone_df, batch_size):
# (definition moved up, no change in content, just ensuring it uses db_config and batch_size from config)

# def insert_cluster_scores_batchwise(username, password, db_config, df_scores, landing_zone_df, batch_size):
# (definition moved up, no change in content, just ensuring it uses db_config and batch_size from config)


def main():
    """Main function to orchestrate the scoring script."""
    config = load_config() # Load configuration first
    db_config = config['database'] # For convenience

    setup_environment()

    username, password, start_date_str, model_score_date_str, end_date_obj = get_user_inputs(config)

    script_logger.info(f"Username: {username}, Start Date: {start_date_str}, Model Score Date: {model_score_date_str}")


    # Initial connection for data fetching (uses main db_config)
    conn_fetch = None 
    try:
        script_logger.info("Connecting to database for data fetching...") # Replaces print
        conn_fetch = connect_db(username, password, db_config, config['logging']['script_logger_name']) 
        df_backup = fetch_data(conn_fetch, start_date_str, model_score_date_str, config['data_fetching']['raw_query_template'], config['logging']['script_logger_name'])
        # script_logger.info(f"Fetched {len(df_backup)} rows.") # This log is now in fetch_data
    except Exception as e:
        script_logger.error(f"Failed to fetch data: {e}", exc_info=True) # Replaces print
        return 
    finally:
        if conn_fetch:
            conn_fetch.close()
            script_logger.info("Database connection for data fetching closed.") # Replaces print

    if df_backup.empty:
        script_logger.warning("No data fetched. Exiting.") # Replaces print
        return

    df = preprocess_data(df_backup, config['logging']['script_logger_name'])
    df = calculate_rolling_metrics(df, config['feature_engineering']['columns_to_smooth'], config['logging']['script_logger_name']) 
    df_filtered = calculate_percentile_ranks(df, end_date_obj, config['logging']['script_logger_name']) 

    if df_filtered.empty: 
        script_logger.warning(f"No data available for scoring date {model_score_date_str} after percentile calculation. Exiting.")
        return

    # Apply models using config
    model_base_path = config['paths']['model_directory']
    script_logger_name_for_apply_model = config['logging']['script_logger_name'] 
    
    df_filtered = apply_model(df_filtered, config['model_features']['core_engagement'], model_base_path, config['model_files']['core_product_engagement'], 'core_product_engagement_score', script_logger_name_for_apply_model)
    df_filtered = apply_model(df_filtered, config['model_features']['core_trend'], model_base_path, config['model_files']['core_product_trend'], 'core_product_trend_score', script_logger_name_for_apply_model)
    df_filtered = apply_model(df_filtered, config['model_features']['additional_engagement'], model_base_path, config['model_files']['additional_feature_engagement'], 'additional_feature_engagement_score', script_logger_name_for_apply_model)
    df_filtered = apply_model(df_filtered, config['model_features']['additional_trend'], model_base_path, config['model_files']['additional_feature_trend'], 'additional_feature_trend_score', script_logger_name_for_apply_model)
    df_filtered = apply_model(df_filtered, config['model_features']['demo'], model_base_path, config['model_files']['demo_score'], 'demo_score', script_logger_name_for_apply_model)
    df_filtered = apply_model(df_filtered, config['model_features']['growth'], model_base_path, config['model_files']['growth_score'], 'cluster_growth_score', script_logger_name_for_apply_model)
    
    script_logger.info("Models applied successfully.") # This was already a logger call
    script_logger.debug("df_filtered head after model application:\n%s", df_filtered.head().to_string()) # This was already a logger call


    score_columns_to_calculate_deciles = {
        "demo_score": "demo_score_decile",
        "core_product_engagement_score": "core_product_engagement_score_decile",
        "core_product_trend_score": "core_product_trend_score_decile",
        "additional_feature_engagement_score": "additional_feature_engagement_score_decile",
        "additional_feature_trend_score": "additional_feature_trend_score_decile",
        "cluster_growth_score": "cluster_growth_decile",
    }
    df_filtered = calculate_deciles(df_filtered, score_columns_to_calculate_deciles, config['logging']['script_logger_name']) 
    script_logger.info("Deciles calculated for cluster scores.") 
    script_logger.debug("df_filtered head after decile calculation:\n%s", df_filtered.head().to_string()) 

    # Plotting for cluster scores (optional)
    # ... (plotting code remains unchanged for now)

    account_scores_df = aggregate_account_scores(
        df_filtered.copy(), 
        "ultimate_parent_account_id", 
        "smoothed_cluster_mrr", 
        "cluster_growth_score", 
        "ds",
        config['logging']['script_logger_name'] 
    )
    
    # Plotting for account scores (optional)
    # ...

    landing_zone_list_df = load_landing_zone_list(
        config['paths']['landing_zone_csv'],
        config['landing_zone_config']['use_cols'],
        config['landing_zone_config']['column_mapping'],
        config['logging']['script_logger_name'] 
    )

    if landing_zone_list_df.empty:
        script_logger.warning("Landing zone list is empty. Halting before database operations.") 
        return
        
    batch_insertion_size = config['database_insertion']['batch_size']
    
    account_table_name_from_config = db_config['account_score_table']
    cluster_table_name_from_config = db_config['cluster_score_table']
    
    # Define SQL for table creation using names from config
    # These definitions are moved here from global scope to ensure they use the loaded config
    account_table_sql_final = f"""
CREATE TABLE IF NOT EXISTS {account_table_name_from_config} (
    ultimate_parent_account_id VARCHAR, weighted_score DOUBLE, smoothed_cluster_mrr DOUBLE,
    account_growth_score DOUBLE, account_growth_decile DOUBLE, processed_date DATE, ds DATE
) WITH (format = 'PARQUET', partitioned_by = ARRAY['ds'])"""

    # The CLUSTER_TABLE_SQL is very long. Instead of duplicating it here,
    # we'll assume the global one is a template or can be formatted.
    # For this exercise, we'll re-define it here for clarity of using config.
    # In a real scenario, this large SQL might be loaded from a separate .sql file or remain a global constant template.
    cluster_table_sql_final = f"""
CREATE TABLE IF NOT EXISTS {cluster_table_name_from_config} (
    cluster_id VARCHAR, group_id VARCHAR, org_id VARCHAR, account_id VARCHAR,
    ultimate_parent_account_id VARCHAR, cluster_age_month INTEGER, cluster_analytics_node_count INTEGER,
    cluster_mdb_major_version DOUBLE, instance_size VARCHAR, is_auto_expand_storage BOOLEAN,
    is_auto_scaling_compute_enabled BOOLEAN, is_auto_scaling_compute_scaledown_enabled BOOLEAN,
    is_backup_selected BOOLEAN, is_sharding BOOLEAN, region_count INTEGER, shard_count INTEGER,
    total_accesses DOUBLE, total_indexes DOUBLE, agg_add_fields_calls DOUBLE, agg_change_stream_calls DOUBLE,
    agg_count_calls DOUBLE, agg_group_calls DOUBLE, agg_limit_calls DOUBLE, agg_lookup_calls DOUBLE,
    agg_match_calls DOUBLE, agg_project_calls DOUBLE, agg_set_calls DOUBLE, agg_skip_calls DOUBLE,
    agg_sort_calls DOUBLE, agg_unwind_calls DOUBLE, agg_total_calls DOUBLE, changestream_events DOUBLE,
    data_size_total_gb DOUBLE, documents_total DOUBLE, reads_writes_per_second_avg DOUBLE,
    collection_count DOUBLE, database_count DOUBLE, cluster_mrr DOUBLE, oa_1d_usage_filesize DOUBLE,
    trigger_events DOUBLE, text_active_1d_usage DOUBLE, text_indexes DOUBLE, vector_active_1d_usage DOUBLE,
    vector_indexes DOUBLE, ts_collection_count DOUBLE, total_hosts_transactions_transactionscollectionwritecount DOUBLE,
    auditlog VARCHAR, adminapi_directapi_usage DOUBLE, instance_size_numeric DOUBLE,
    smoothed_cluster_mrr DOUBLE, smoothed_reads_writes_per_second_avg DOUBLE, smoothed_data_size_total_gb DOUBLE,
    smoothed_documents_total DOUBLE, smoothed_database_count DOUBLE, smoothed_collection_count DOUBLE,
    smoothed_total_accesses DOUBLE, smoothed_total_indexes DOUBLE, smoothed_auditlog DOUBLE,
    smoothed_agg_total_calls DOUBLE, smoothed_agg_set_calls DOUBLE, smoothed_agg_match_calls DOUBLE,
    smoothed_agg_group_calls DOUBLE, smoothed_agg_project_calls DOUBLE, smoothed_agg_limit_calls DOUBLE,
    smoothed_agg_sort_calls DOUBLE, smoothed_agg_unwind_calls DOUBLE, smoothed_agg_add_fields_calls DOUBLE,
    smoothed_agg_lookup_calls DOUBLE, smoothed_agg_count_calls DOUBLE, smoothed_agg_skip_calls DOUBLE,
    smoothed_cluster_analytics_node_count DOUBLE, smoothed_adminapi_directapi_usage DOUBLE,
    smoothed_agg_change_stream_calls DOUBLE, smoothed_changestream_events DOUBLE,
    smoothed_total_hosts_transactions_transactionscollectionwritecount DOUBLE, smoothed_oa_1d_usage_filesize DOUBLE,
    smoothed_text_indexes DOUBLE, smoothed_text_active_1d_usage DOUBLE, smoothed_ts_collection_count DOUBLE,
    smoothed_trigger_events DOUBLE, smoothed_vector_indexes DOUBLE, smoothed_vector_active_1d_usage DOUBLE,
    smoothed_cluster_mrr_30d_pct_change DOUBLE, smoothed_reads_writes_per_second_avg_30d_pct_change DOUBLE,
    smoothed_data_size_total_gb_30d_pct_change DOUBLE, smoothed_documents_total_30d_pct_change DOUBLE,
    smoothed_database_count_30d_pct_change DOUBLE, smoothed_collection_count_30d_pct_change DOUBLE,
    smoothed_total_accesses_30d_pct_change DOUBLE, smoothed_total_indexes_30d_pct_change DOUBLE,
    smoothed_auditlog_30d_pct_change DOUBLE, smoothed_agg_total_calls_30d_pct_change DOUBLE,
    smoothed_agg_set_calls_30d_pct_change DOUBLE, smoothed_agg_match_calls_30d_pct_change DOUBLE,
    smoothed_agg_group_calls_30d_pct_change DOUBLE, smoothed_agg_project_calls_30d_pct_change DOUBLE,
    smoothed_agg_limit_calls_30d_pct_change DOUBLE, smoothed_agg_sort_calls_30d_pct_change DOUBLE,
    smoothed_agg_unwind_calls_30d_pct_change DOUBLE, smoothed_agg_add_fields_calls_30d_pct_change DOUBLE,
    smoothed_agg_lookup_calls_30d_pct_change DOUBLE, smoothed_agg_count_calls_30d_pct_change DOUBLE,
    smoothed_agg_skip_calls_30d_pct_change DOUBLE, smoothed_cluster_analytics_node_count_30d_pct_change DOUBLE,
    smoothed_adminapi_directapi_usage_30d_pct_change DOUBLE, smoothed_agg_change_stream_calls_30d_pct_change DOUBLE,
    smoothed_changestream_events_30d_pct_change DOUBLE,
    smoothed_total_hosts_transactions_transactionscollectionwritecount_30d_pct_change DOUBLE,
    smoothed_oa_1d_usage_filesize_30d_pct_change DOUBLE, smoothed_text_indexes_30d_pct_change DOUBLE,
    smoothed_text_active_1d_usage_30d_pct_change DOUBLE, smoothed_ts_collection_count_30d_pct_change DOUBLE,
    smoothed_trigger_events_30d_pct_change DOUBLE, smoothed_vector_indexes_30d_pct_change DOUBLE,
    smoothed_vector_active_1d_usage_30d_pct_change DOUBLE,
    smoothed_cluster_mrr_percentile_by_instance DOUBLE, smoothed_reads_writes_per_second_avg_percentile_by_instance DOUBLE,
    smoothed_data_size_total_gb_percentile_by_instance DOUBLE, smoothed_documents_total_percentile_by_instance DOUBLE,
    smoothed_database_count_percentile_by_instance DOUBLE, smoothed_collection_count_percentile_by_instance DOUBLE,
    smoothed_total_accesses_percentile_by_instance DOUBLE, smoothed_total_indexes_percentile_by_instance DOUBLE,
    smoothed_auditlog_percentile_by_instance DOUBLE, smoothed_agg_total_calls_percentile_by_instance DOUBLE,
    smoothed_agg_set_calls_percentile_by_instance DOUBLE, smoothed_agg_match_calls_percentile_by_instance DOUBLE,
    smoothed_agg_group_calls_percentile_by_instance DOUBLE, smoothed_agg_project_calls_percentile_by_instance DOUBLE,
    smoothed_agg_limit_calls_percentile_by_instance DOUBLE, smoothed_agg_sort_calls_percentile_by_instance DOUBLE,
    smoothed_agg_unwind_calls_percentile_by_instance DOUBLE, smoothed_agg_add_fields_calls_percentile_by_instance DOUBLE,
    smoothed_agg_lookup_calls_percentile_by_instance DOUBLE, smoothed_agg_count_calls_percentile_by_instance DOUBLE,
    smoothed_agg_skip_calls_percentile_by_instance DOUBLE,
    smoothed_cluster_analytics_node_count_percentile_by_instance DOUBLE,
    smoothed_adminapi_directapi_usage_percentile_by_instance DOUBLE,
    smoothed_agg_change_stream_calls_percentile_by_instance DOUBLE, smoothed_changestream_events_percentile_by_instance DOUBLE,
    smoothed_total_hosts_transactions_transactionscollectionwritecount_percentile_by_instance DOUBLE,
    smoothed_oa_1d_usage_filesize_percentile_by_instance DOUBLE, smoothed_text_indexes_percentile_by_instance DOUBLE,
    smoothed_text_active_1d_usage_percentile_by_instance DOUBLE, smoothed_ts_collection_count_percentile_by_instance DOUBLE,
    smoothed_trigger_events_percentile_by_instance DOUBLE, smoothed_vector_indexes_percentile_by_instance DOUBLE,
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
    core_product_engagement_score DOUBLE, core_product_trend_score DOUBLE,
    additional_feature_engagement_score DOUBLE, additional_feature_trend_score DOUBLE, demo_score DOUBLE,
    cluster_growth_score DOUBLE, demo_score_decile INTEGER, core_product_engagement_score_decile INTEGER,
    core_product_trend_score_decile INTEGER, additional_feature_engagement_score_decile INTEGER,
    additional_feature_trend_score_decile INTEGER, cluster_growth_decile INTEGER, weighted_score DOUBLE,
    processed_date DATE, ds DATE
) WITH (format = 'PARQUET', partitioned_by = ARRAY['ds'])"""


    script_logger.info("Starting database table creation and data insertion...") # Was print
    create_table(username, password, db_config, account_table_sql_final, 'account_score_table', config['logging']['script_logger_name'])
    create_table(username, password, db_config, cluster_table_sql_final, 'cluster_score_table', config['logging']['script_logger_name']) 

    insert_account_scores_batchwise(username, password, db_config, account_scores_df, landing_zone_list_df, batch_insertion_size, config['logging']['script_logger_name'])
    insert_cluster_scores_batchwise(username, password, db_config, df_filtered, landing_zone_list_df, batch_insertion_size, config['logging']['script_logger_name'])
    
    script_logger.info("Scoring script execution completed.") # Was print
    if not df_filtered.empty:
        script_logger.info(f"Final cluster scores target date: {df_filtered['ds'].iloc[0]}") # Was print
    if not account_scores_df.empty:
        script_logger.info(f"Final account scores target date (ds): {account_scores_df['ds'].iloc[0]}") # Was print
        script_logger.info(f"Final account scores processed date: {account_scores_df['processed_date'].iloc[0]}") # Was print


if __name__ == "__main__":
    main()



