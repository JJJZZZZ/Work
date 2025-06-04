import unittest
import pandas as pd
import numpy as np
from core.data_processor import DataProcessor # Assuming it's in 'core'

class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = DataProcessor()

    def _create_test_df(self, num_rows: int, num_cols: int = 5) -> pd.DataFrame:
        """Helper function to create a DataFrame with mixed data types."""
        data = {
            f'col_int_{i}': np.random.randint(0, 100, size=num_rows) for i in range(num_cols // 2)
        }
        for i in range(num_cols // 2, num_cols):
            data[f'col_str_{i}'] = [f'value_{j}' for j in range(num_rows)]

        # Ensure at least one numeric column for stats calculation
        if num_cols // 2 == 0 and num_cols > 0 :
             data['col_int_0'] = np.random.randint(0, 100, size=num_rows)

        if not data: # Handle case for num_cols = 0, though unlikely for summary
            return pd.DataFrame()

        return pd.DataFrame(data)

    def test_get_data_summary_with_sampling(self):
        """Test get_data_summary with a large dataset where sampling should occur."""
        large_df_rows = 2000
        sample_size = 500
        df_large = self._create_test_df(large_df_rows)

        summary = self.processor.get_data_summary(df_large, sample_large_datasets=True, sample_size=sample_size)

        self.assertIn('notes', summary, "Summary should contain 'notes' when sampling occurs.")
        self.assertTrue(any(f"sample of {sample_size} rows" in note for note in summary['notes']))

        self.assertEqual(summary['shape'][0], large_df_rows, "summary['shape'][0] should reflect original row count.")
        self.assertIn('display_shape', summary, "Summary should have 'display_shape' when sampling might occur.")
        self.assertEqual(summary['display_shape'][0], sample_size, "summary['display_shape'][0] should reflect sample_size.")

        self.assertIn('numeric_stats', summary, "Summary should include numeric_stats.")
        if summary['numeric_columns']: # Check if there are numeric columns to have stats for
            # Check that stats are based on the sampled data, e.g. count should be sample_size
            first_numeric_col = summary['numeric_columns'][0]
            self.assertEqual(summary['numeric_stats'][first_numeric_col]['count'], sample_size)

    def test_get_data_summary_without_sampling_large_dataset_sampling_disabled(self):
        """Test get_data_summary with a large dataset but sampling disabled."""
        large_df_rows = 2000
        df_large = self._create_test_df(large_df_rows)

        summary = self.processor.get_data_summary(df_large, sample_large_datasets=False)

        self.assertNotIn('notes', summary, "Summary should not contain 'notes' if sampling is disabled.")
        self.assertEqual(summary['shape'][0], large_df_rows, "summary['shape'][0] should reflect original row count.")
        # 'display_shape' should still exist and match 'shape'
        self.assertIn('display_shape', summary)
        self.assertEqual(summary['display_shape'][0], large_df_rows)

        self.assertIn('numeric_stats', summary)
        if summary['numeric_columns']:
            first_numeric_col = summary['numeric_columns'][0]
            self.assertEqual(summary['numeric_stats'][first_numeric_col]['count'], large_df_rows)

    def test_get_data_summary_without_sampling_small_dataset(self):
        """Test get_data_summary with a small dataset where sampling should not occur by default."""
        small_df_rows = 300
        default_sample_size = 1000 # Default sample_size in DataProcessor.get_data_summary
        df_small = self._create_test_df(small_df_rows)

        summary = self.processor.get_data_summary(df_small) # Rely on default sample_large_datasets=True

        self.assertNotIn('notes', summary, "Summary should not contain 'notes' for small datasets.")
        self.assertEqual(summary['shape'][0], small_df_rows, "summary['shape'][0] should reflect original row count.")
        self.assertIn('display_shape', summary)
        self.assertEqual(summary['display_shape'][0], small_df_rows)

        self.assertIn('numeric_stats', summary)
        if summary['numeric_columns']:
            first_numeric_col = summary['numeric_columns'][0]
            self.assertEqual(summary['numeric_stats'][first_numeric_col]['count'], small_df_rows)

    def test_get_data_summary_edge_case_empty_df(self):
        """Test get_data_summary with an empty DataFrame."""
        df_empty = pd.DataFrame()
        summary = self.processor.get_data_summary(df_empty)
        self.assertEqual(summary['shape'][0], 0)
        self.assertEqual(summary['shape'][1], 0)
        self.assertEqual(summary['display_shape'][0], 0)
        self.assertEqual(summary['display_shape'][1], 0)
        self.assertEqual(len(summary['columns']), 0)
        self.assertNotIn('notes', summary)

    def test_get_data_summary_edge_case_one_row_df(self):
        """Test get_data_summary with a DataFrame containing only one row."""
        df_one_row = self._create_test_df(num_rows=1)
        summary = self.processor.get_data_summary(df_one_row, sample_size=500) # sample_size > num_rows
        self.assertEqual(summary['shape'][0], 1)
        self.assertEqual(summary['display_shape'][0], 1)
        self.assertNotIn('notes', summary)
        if summary['numeric_columns']:
            first_numeric_col = summary['numeric_columns'][0]
            self.assertEqual(summary['numeric_stats'][first_numeric_col]['count'], 1)

if __name__ == '__main__':
    unittest.main()
