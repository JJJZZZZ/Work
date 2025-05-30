import unittest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
import numpy as np
import yaml
from datetime import datetime

# Assuming scoring_auto.py is in the same directory or accessible via PYTHONPATH
# We will import functions as needed to avoid issues with partially modified/non-existent functions
# from scoring_auto import load_config, preprocess_data, calculate_rolling_metrics, calculate_percentile_ranks, calculate_deciles, aggregate_account_scores

# Dummy config for testing purposes, as functions might rely on config structure for logging
dummy_config_for_logging = {
    'logging': {
        'script_logger_name': 'test_script_logger'
    }
}

# Mocking logger for functions that expect script_logger_name
# This avoids needing to set up the full logging apparatus for unit tests
# and allows functions to call logging.getLogger without error.
# We can assert that logger methods were called if needed.
mock_logger = MagicMock()

class TestConfigLoading(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open, read_data="database:\n  host: test_host\n  port: 123\nlogging:\n  log_file: test.log\n  level: DEBUG\n  format: '%(message)s'\n  script_logger_name: 'TestScoringScript'")
    @patch("scoring_auto.yaml.safe_load")
    def test_load_config_success(self, mock_safe_load, mock_file_open):
        # Need to import load_config here to ensure it uses the patched open
        from scoring_auto import load_config
        
        expected_config = {
            "database": {"host": "test_host", "port": 123},
            "logging": {"log_file": "test.log", "level": "DEBUG", "format": "%(message)s", "script_logger_name": "TestScoringScript"}
        }
        mock_safe_load.return_value = expected_config
        
        config = load_config("dummy_path.yaml")
        
        mock_file_open.assert_called_once_with("dummy_path.yaml", 'r')
        mock_safe_load.assert_called_once()
        self.assertEqual(config, expected_config)

    @patch("builtins.open", side_effect=FileNotFoundError("File not found"))
    def test_load_config_file_not_found(self, mock_file_open):
        from scoring_auto import load_config
        with self.assertRaises(SystemExit) as cm:
            load_config("non_existent.yaml")
        self.assertIn("CRITICAL: Configuration file non_existent.yaml not found.", str(cm.exception))

    @patch("builtins.open", new_callable=mock_open, read_data="database: host: test_host\nport; 123") # Invalid YAML
    @patch("scoring_auto.yaml.safe_load", side_effect=yaml.YAMLError("YAML parsing error"))
    def test_load_config_yaml_error(self, mock_safe_load, mock_file_open):
        from scoring_auto import load_config
        with self.assertRaises(SystemExit) as cm:
            load_config("invalid_yaml.yaml")
        self.assertIn("CRITICAL: Error parsing YAML configuration", str(cm.exception))


class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # A dummy logger name for functions that require it
        self.dummy_logger_name = "test_logger"
        # Mock the logger instance that would be created by getLogger
        # This avoids errors if functions try to use the logger, but doesn't test logging content itself
        self.patcher = patch('scoring_auto.logging.getLogger', return_value=mock_logger)
        self.mock_get_logger = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        mock_logger.reset_mock() # Reset call counts etc. for each test

    def test_preprocess_data(self):
        from scoring_auto import preprocess_data
        sample_data = {
            'ds': ['2023-01-01', '2023-01-02'],
            'instance_size': ['M10', 'M20'],
            'cluster_mdb_major_version': ['4.4', '5.0']
        }
        df_input = pd.DataFrame(sample_data)
        
        # Expected transformations
        df_expected = df_input.copy()
        df_expected['ds'] = pd.to_datetime(df_expected['ds'])
        df_expected['instance_size_numeric'] = [10.0, 20.0]
        df_expected['cluster_mdb_major_version'] = [4.4, 5.0]

        # Pass the dummy logger name
        df_result = preprocess_data(df_input.copy(), self.dummy_logger_name)
        
        pd.testing.assert_series_equal(df_result['ds'], df_expected['ds'], check_dtype=False)
        pd.testing.assert_series_equal(df_result['instance_size_numeric'], df_expected['instance_size_numeric'], check_dtype=False)
        pd.testing.assert_series_equal(df_result['cluster_mdb_major_version'], df_expected['cluster_mdb_major_version'], check_dtype=False)

    def test_calculate_rolling_metrics(self):
        from scoring_auto import calculate_rolling_metrics
        data = {
            'cluster_id': ['A', 'A', 'A', 'B', 'B', 'B'],
            'ds': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-01', '2023-01-02', '2023-01-03']),
            'cluster_mrr': [10, 20, 30, 100, 100, 100],
            'reads_writes_per_second_avg': [1, 2, 3, 10, 10, 10]
        }
        df = pd.DataFrame(data)
        # Reduce window for easier testing if necessary, or use min_periods
        # For now, assume default window=30, min_periods=1
        
        # This function needs the list of columns to smooth
        # We can mock the config or pass it directly if the function is adapted
        # For now, let's use a simplified list based on the input data
        columns_to_smooth = ['cluster_mrr', 'reads_writes_per_second_avg']

        df_result = calculate_rolling_metrics(df.copy(), columns_to_smooth, self.dummy_logger_name)

        self.assertTrue('smoothed_cluster_mrr' in df_result.columns)
        self.assertTrue('smoothed_reads_writes_per_second_avg' in df_result.columns)
        self.assertTrue('smoothed_cluster_mrr_30d_pct_change' in df_result.columns)
        
        # Spot check some values for cluster A (window=30, min_periods=1)
        # smoothed_cluster_mrr for A: [10.0, 15.0, 20.0]
        self.assertAlmostEqual(df_result[df_result['cluster_id'] == 'A']['smoothed_cluster_mrr'].iloc[0], 10.0)
        self.assertAlmostEqual(df_result[df_result['cluster_id'] == 'A']['smoothed_cluster_mrr'].iloc[1], 15.0)
        self.assertAlmostEqual(df_result[df_result['cluster_id'] == 'A']['smoothed_cluster_mrr'].iloc[2], 20.0)
        
        # pct_change will be NaN for first 30 periods (or less if less data)
        # With min_periods=1 and periods=30 for pct_change, it will be NaN if less than 31 data points for the group.
        # Here, with only 3 points, pct_change will be all NaN.
        self.assertTrue(df_result[df_result['cluster_id'] == 'A']['smoothed_cluster_mrr_30d_pct_change'].isna().all())


    def test_calculate_percentile_ranks(self):
        from scoring_auto import calculate_percentile_ranks
        data = {
            'ds': pd.to_datetime(['2023-01-30', '2023-01-30', '2023-01-30', '2023-01-30']),
            'instance_size': ['M10', 'M10', 'M20', 'M20'],
            'smoothed_metric_A': [10, 20, 5, 15],
            'smoothed_metric_B': [100, 50, 200, 250]
        }
        df = pd.DataFrame(data)
        end_date = datetime(2023, 1, 30)

        # The original print statement needs to be converted to logging or mocked.
        # For now, we assume it's converted or our global mock_logger handles it.
        df_result = calculate_percentile_ranks(df.copy(), end_date, self.dummy_logger_name)

        self.assertTrue('smoothed_metric_A_percentile_by_instance' in df_result.columns)
        self.assertTrue('smoothed_metric_B_percentile_by_instance' in df_result.columns)

        # Expected percentiles for M10: 10 is 0.5, 20 is 1.0
        # Expected percentiles for M20: 5 is 0.5, 15 is 1.0
        expected_A_percentiles = [0.5, 1.0, 0.5, 1.0] 
        self.assertListEqual(df_result['smoothed_metric_A_percentile_by_instance'].tolist(), expected_A_percentiles)

    def test_calculate_deciles(self):
        from scoring_auto import calculate_deciles
        data = {'score': np.linspace(0, 0.99, 100)} # 100 values for easy deciles
        df = pd.DataFrame(data)
        
        score_map = {'score': 'score_decile'}
        df_result = calculate_deciles(df.copy(), score_map, self.dummy_logger_name)
        
        self.assertTrue('score_decile' in df_result.columns)
        # Check if deciles range from 1 to 10
        self.assertEqual(df_result['score_decile'].min(), 1)
        self.assertEqual(df_result['score_decile'].max(), 10)
        self.assertEqual(len(df_result[df_result['score_decile'] == 1]), 10) # Each decile should have 10 members
        self.assertEqual(len(df_result[df_result['score_decile'] == 10]), 10)

    def test_aggregate_account_scores(self):
        # This function calls calculate_deciles internally.
        # We'll test its aggregation logic primarily.
        # Need to import/define calculate_deciles if it's not already available.
        from scoring_auto import aggregate_account_scores, calculate_deciles 

        data = {
            'ultimate_parent_account_id': ['acc1', 'acc1', 'acc2', 'acc2', 'acc3'],
            'cluster_growth_score':       [0.1,    0.2,    0.8,    0.9,    0.5 ], # Raw scores
            'smoothed_cluster_mrr':       [100,    100,    50,     50,     200 ], # MRRs
            'ds': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03'])
        }
        df_filtered = pd.DataFrame(data)

        # Mock calculate_deciles or ensure it's robust from previous test
        # For simplicity, we rely on the previously tested calculate_deciles
        
        account_scores_df = aggregate_account_scores(
            df_filtered.copy(), 
            "ultimate_parent_account_id", 
            "smoothed_cluster_mrr", 
            "cluster_growth_score", 
            "ds",
            self.dummy_logger_name # Pass dummy logger name
        )
        
        self.assertEqual(len(account_scores_df), 3) # 3 unique accounts
        
        # Check acc1: weighted_score = (0.1*100 + 0.2*100) = 10 + 20 = 30. mrr_sum = 200. score = 30/200 = 0.15
        acc1_score = account_scores_df[account_scores_df['ultimate_parent_account_id'] == 'acc1']['account_growth_score'].iloc[0]
        self.assertAlmostEqual(acc1_score, 0.15) 
        
        # Check acc2: weighted_score = (0.8*50 + 0.9*50) = 40 + 45 = 85. mrr_sum = 100. score = 85/100 = 0.85
        acc2_score = account_scores_df[account_scores_df['ultimate_parent_account_id'] == 'acc2']['account_growth_score'].iloc[0]
        self.assertAlmostEqual(acc2_score, 0.85)

        # Check acc3: weighted_score = 0.5*200 = 100. mrr_sum = 200. score = 100/200 = 0.5
        acc3_score = account_scores_df[account_scores_df['ultimate_parent_account_id'] == 'acc3']['account_growth_score'].iloc[0]
        self.assertAlmostEqual(acc3_score, 0.5)

        self.assertTrue('account_growth_decile' in account_scores_df.columns)
        self.assertTrue('processed_date' in account_scores_df.columns)


if __name__ == '__main__':
    unittest.main()
