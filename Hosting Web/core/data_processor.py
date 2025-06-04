"""
Data Processing Module for Propensity Matching Tool
Handles data validation, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Handles all data processing tasks for propensity matching"""
    
    def __init__(self):
        self.scaler = None
    
    def _convert_to_serializable(self, obj):
        """Convert numpy/pandas types to native Python types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isinf(obj) or np.isnan(obj):
                return None  # Or str(obj) if "Infinity" or "NaN" strings are preferred
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
        
    def validate_dataframe(self, df: pd.DataFrame, 
                          primary_identifier: str, 
                          treatment_flag: str) -> Dict[str, bool]:
        """
        Validate the dataframe for basic requirements
        
        Args:
            df: Input dataframe
            primary_identifier: Column name for primary identifier
            treatment_flag: Column name for treatment flag
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'has_primary_identifier': primary_identifier in df.columns,
            'has_treatment_flag': treatment_flag in df.columns,
            'has_data': len(df) > 0,
            'treatment_flag_binary': False,
            'no_all_null_columns': True,
            'primary_id_unique': False
        }
        
        if validation_results['has_treatment_flag']:
            unique_values = df[treatment_flag].dropna().unique()
            validation_results['treatment_flag_binary'] = len(unique_values) == 2 and set(unique_values).issubset({0, 1, True, False})
        
        if validation_results['has_primary_identifier']:
            validation_results['primary_id_unique'] = df[primary_identifier].nunique() == len(df)
        
        # Check for columns that are all null
        all_null_columns = df.columns[df.isnull().all()].tolist()
        validation_results['no_all_null_columns'] = len(all_null_columns) == 0
        
        return validation_results
    
    def prepare_data_for_matching(self, 
                                 df: pd.DataFrame,
                                 primary_identifier: str,
                                 columns_for_matching: List[str],
                                 ds_month_column: str,
                                 treatment_flag: str) -> pd.DataFrame:
        """
        Prepare data by selecting relevant columns and creating treatment flag
        
        Args:
            df: Input dataframe
            primary_identifier: Primary identifier column
            columns_for_matching: List of columns to use for matching
            ds_month_column: Date/month column
            treatment_flag: Treatment flag column
            
        Returns:
            Processed dataframe
        """
        # Select required columns
        required_columns = [primary_identifier, ds_month_column, treatment_flag] + columns_for_matching
        
        # Remove duplicates from required_columns
        required_columns = list(dict.fromkeys(required_columns))
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        
        df_selected = df[required_columns].copy()
        
        # If the treatment flag column is not named "treatment_flag", rename it
        if treatment_flag != "treatment_flag":
            df_selected = df_selected.rename(columns={treatment_flag: "treatment_flag"})
        
        # Ensure treatment_flag is binary (0/1)
        if df_selected["treatment_flag"].dtype == bool:
            df_selected["treatment_flag"] = df_selected["treatment_flag"].astype(int)
        elif df_selected["treatment_flag"].dtype == 'object':
            # Handle string values that might represent boolean
            df_selected["treatment_flag"] = df_selected["treatment_flag"].map(
                lambda x: 1 if str(x).lower() in ['true', '1', 'yes', 'treatment'] else 0
            )
        
        # Verify treatment flag is binary
        unique_values = df_selected["treatment_flag"].dropna().unique()
        if not set(unique_values).issubset({0, 1}):
            raise ValueError(f"Treatment flag must be binary (0/1), got values: {unique_values}")
        
        return df_selected
    
    def handle_missing_values(self, 
                             df: pd.DataFrame, 
                             method: str = 'drop',
                             exclude_columns: List[str] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataframe
        
        Args:
            df: Input dataframe
            method: Method to handle missing values ('drop', 'fill_mean', 'fill_median')
            exclude_columns: Columns to exclude from processing
            
        Returns:
            Dataframe with missing values handled
        """
        df_processed = df.copy()
        
        if exclude_columns is None:
            exclude_columns = []
        
        # Get columns to process
        cols_to_process = [col for col in df.columns if col not in exclude_columns]
        
        if method == 'drop':
            df_processed = df_processed.dropna(subset=cols_to_process)
        elif method == 'fill_mean':
            numeric_cols = df_processed[cols_to_process].select_dtypes(include=["number"]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
        elif method == 'fill_median':
            numeric_cols = df_processed[cols_to_process].select_dtypes(include=["number"]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
        elif method == 'fill_mode':
            for col in cols_to_process:
                if df_processed[col].isnull().any():
                    mode_value = df_processed[col].mode()
                    if len(mode_value) > 0:
                        df_processed[col] = df_processed[col].fillna(mode_value[0])
        
        return df_processed
    
    def handle_outliers(self, 
                       df: pd.DataFrame, 
                       numeric_columns: List[str],
                       method: str = 'winsorize') -> pd.DataFrame:
        """
        Handle outliers in numeric columns
        
        Args:
            df: Input dataframe
            numeric_columns: List of numeric columns to process
            method: Method to handle outliers ('winsorize', 'remove')
            
        Returns:
            Dataframe with outliers handled
        """
        df_processed = df.copy()
        
        for col in numeric_columns:
            if col in df_processed.columns:
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                if method == 'winsorize':
                    # Cap values at bounds
                    df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
                elif method == 'remove':
                    # Remove rows with outliers
                    df_processed = df_processed[
                        (df_processed[col] >= lower_bound) & 
                        (df_processed[col] <= upper_bound)
                    ]
        
        return df_processed
    
    def standardize_numeric_columns(self, 
                                   df: pd.DataFrame, 
                                   numeric_columns: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Standardize numeric columns using StandardScaler
        
        Args:
            df: Input dataframe
            numeric_columns: List of numeric columns to standardize
            
        Returns:
            Tuple of (processed dataframe, fitted scaler)
        """
        df_processed = df.copy()
        scaler = StandardScaler()
        
        # Filter to only columns that exist in the dataframe
        existing_numeric_columns = [col for col in numeric_columns if col in df_processed.columns]
        
        if existing_numeric_columns:
            df_processed[existing_numeric_columns] = scaler.fit_transform(df_processed[existing_numeric_columns])
            self.scaler = scaler
        
        return df_processed, scaler
    
    def sample_control_group(self, 
                           df: pd.DataFrame,
                           primary_identifier: str,
                           sample_fraction: float = 0.5,
                           random_state: int = 42) -> pd.DataFrame:
        """
        Sample the control group to reduce computation time
        
        Args:
            df: Input dataframe
            primary_identifier: Primary identifier column
            sample_fraction: Fraction of control group to sample
            random_state: Random state for reproducibility
            
        Returns:
            Dataframe with sampled control group
        """
        treatment_df = df[df['treatment_flag'] == 1].copy()
        control_df = df[df['treatment_flag'] == 0].copy()
        
        # Sample unique control IDs
        unique_control_ids = control_df[primary_identifier].unique()
        sampled_ids = np.random.choice(
            unique_control_ids, 
            size=int(len(unique_control_ids) * sample_fraction), 
            replace=False
        )
        
        # Filter control group to sampled IDs
        sampled_control_df = control_df[control_df[primary_identifier].isin(sampled_ids)]
        
        # Combine treatment and sampled control
        result_df = pd.concat([treatment_df, sampled_control_df], ignore_index=True)
        
        return result_df
    
    def get_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Get numeric and string column types
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (numeric_columns, string_columns)
        """
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        string_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove treatment_flag from numeric columns since it's categorical
        if "treatment_flag" in numeric_columns:
            numeric_columns.remove("treatment_flag")
        
        return numeric_columns, string_columns
    
    def get_data_summary(self, df: pd.DataFrame,
                         sample_large_datasets: bool = True,
                         sample_size: int = 1000) -> Dict:
        """
        Get comprehensive data summary, with an option to sample large datasets.
        
        Args:
            df: Input dataframe
            sample_large_datasets: If True, sample df if rows > sample_size.
            sample_size: Number of rows to sample if df is large.
            
        Returns:
            Dictionary with data summary statistics (JSON serializable)
        """
        print(f"DEBUG: get_data_summary started for dataframe shape: {df.shape}")
        
        # df_display will be the DataFrame used for generating most of the summary statistics.
        # It can be the original DataFrame or a sample of it.
        df_display = df
        summary_notes = [] # To store any notes regarding the summary, e.g., if it's sampled.

        # Sampling logic:
        # If sample_large_datasets is True and the DataFrame has more rows than sample_size,
        # a random sample of the DataFrame is taken for display and summary calculation.
        # This improves performance for very large datasets.
        if sample_large_datasets and len(df) > sample_size:
            print(f"DEBUG: DataFrame is large (rows: {len(df)} > sample_size: {sample_size}). Sampling...")
            # Take a random sample. random_state is used for reproducibility.
            df_display = df.sample(n=sample_size, random_state=42)
            summary_notes.append(f"Statistics based on a sample of {sample_size} rows from the original {len(df)} rows.")
            print(f"DEBUG: Sampled DataFrame shape: {df_display.shape}")
        else:
            # If not sampling, df_display remains the original DataFrame.
            print(f"DEBUG: Using full DataFrame for summary (rows: {len(df)}).")

        # All subsequent calculations for the summary should use df_display.
        # Retrieve column types from df_display.
        numeric_cols, string_cols = self.get_column_types(df_display)
        print(f"DEBUG: Column types identified (from df_display) - numeric: {len(numeric_cols)}, string: {len(string_cols)}")
        
        # Create summary with proper type conversion.
        print(f"DEBUG: Creating basic summary...")
        summary = {
            'shape': [int(df.shape[0]), int(df.shape[1])],  # Store the original shape of the input DataFrame.
            'display_shape': [int(df_display.shape[0]), int(df_display.shape[1])], # Shape of data used for generating stats (could be sampled).
            'columns': df_display.columns.tolist(), # Columns from df_display.
            'numeric_columns': numeric_cols, # Numeric columns from df_display.
            'string_columns': string_cols,   # String columns from df_display.
            # Missing values calculated on df_display.
            'missing_values': {col: int(df_display[col].isnull().sum()) for col in df_display.columns},
            'missing_percentages': {
                col: None if pd.isna(df_display[col].isnull().mean()) else float(round(df_display[col].isnull().mean() * 100, 2))
                for col in df_display.columns
            },
            'data_types': {col: str(df_display[col].dtype) for col in df_display.columns}, # Data types from df_display.
            # Memory usage of the DataFrame used for display/summary.
            'memory_usage_display': int(df_display.memory_usage(deep=True).sum()),
            # Duplicated rows count in the DataFrame used for display/summary.
            'duplicated_rows_display': int(df_display.duplicated().sum())
        }
        # If any notes were added (e.g., about sampling), include them in the summary.
        if summary_notes:
            summary['notes'] = summary_notes

        print(f"DEBUG: Basic summary created")
        
        # Add descriptive statistics for numeric columns from df_display.
        if numeric_cols:
            print(f"DEBUG: Computing statistics for {len(numeric_cols)} numeric columns (from df_display)...")
            desc_stats = df_display[numeric_cols].describe() # Calculated on df_display.
            summary['numeric_stats'] = {}
            for col in numeric_cols:
                summary['numeric_stats'][col] = {
                    # Ensure stats are JSON serializable and handle NaNs.
                    stat: float(desc_stats.loc[stat, col]) if not pd.isna(desc_stats.loc[stat, col]) else None
                    for stat in desc_stats.index
                }
            print(f"DEBUG: Numeric statistics computed")
        
        # Ensure all values are JSON serializable
        print(f"DEBUG: Converting to JSON serializable format...")
        summary = self._convert_to_serializable(summary)
        print(f"DEBUG: Conversion completed, returning summary")
        
        return summary 