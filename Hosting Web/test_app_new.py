import unittest
import pandas as pd
import numpy as np # Make sure numpy is imported
import io
import os
from fastapi.testclient import TestClient
from app import app # Import your FastAPI app instance

# Ensure the app uses the DataProcessor with the new summary logic.

class TestApp(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        # Default sample_size in DataProcessor.get_data_summary is 1000
        # This is important for asserting whether sampling notes appear.
        self.default_summary_sample_size = 1000

    def _create_csv_file(self, num_rows: int, num_cols: int = 5, filename: str = "test.csv") -> str:
        """Helper function to create a temporary CSV file and return its path."""
        data = {}
        for i in range(num_cols):
            data[f'col_{i}'] = np.random.rand(num_rows) # Use np for data generation
        df = pd.DataFrame(data)
        # Ensure the directory exists if filename includes a path, though not here.
        df.to_csv(filename, index=False)
        return filename

    def tearDown(self):
        """Clean up any files created during tests."""
        files_to_remove = ["test_large.csv", "test_small.csv", "test.csv", "test_empty.csv", "test_invalid.bin"] # Changed test_invalid.txt to .bin
        for f in files_to_remove:
            if os.path.exists(f):
                os.remove(f)

    def test_health_check(self):
        """Test the /health endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("version", data)

    def test_upload_data_large_csv_expect_sampling(self):
        """Test /upload-data with a large CSV that should trigger sampling in summary."""
        num_rows = self.default_summary_sample_size + 500
        num_cols = 5
        test_file_path = self._create_csv_file(num_rows, num_cols, "test_large.csv")

        with open(test_file_path, 'rb') as f:
            response = self.client.post("/upload-data", files={"file": ("test_large.csv", f, "text/csv")})

        self.assertEqual(response.status_code, 200, f"Response content: {response.content}")
        result = response.json()
        self.assertTrue(result['success'])
        summary = result['summary']

        self.assertIn('notes', summary, "Summary should contain 'notes' when sampling occurs for large CSV.")
        self.assertTrue(any(f"sample of {self.default_summary_sample_size}" in note for note in summary['notes']),
                        f"Notes should indicate sampling at {self.default_summary_sample_size} rows. Got: {summary.get('notes')}")
        self.assertEqual(summary['shape'][0], num_rows, "summary['shape'][0] should reflect original row count.")
        self.assertEqual(summary['display_shape'][0], self.default_summary_sample_size,
                         "summary['display_shape'][0] should reflect the sample size.")

    def test_upload_data_very_large_csv_chunked_reading_check(self):
        """Test /upload-data with a very large CSV to ensure it's processed (chunking is internal)."""
        num_rows = 25000 # Typically larger than default chunk_size in app.py (e.g. 10000)
        num_cols = 3
        test_file_path = self._create_csv_file(num_rows, num_cols, "test_large.csv")

        with open(test_file_path, 'rb') as f:
            response = self.client.post("/upload-data", files={"file": ("test_large.csv", f, "text/csv")})

        self.assertEqual(response.status_code, 200, f"Response content: {response.content}")
        result = response.json()
        self.assertTrue(result['success'])
        summary = result['summary']
        self.assertEqual(summary['shape'][0], num_rows)

        if num_rows > self.default_summary_sample_size:
            self.assertIn('notes', summary, f"Notes missing for large file. Summary: {summary}")
            self.assertTrue(any(f"sample of {self.default_summary_sample_size}" in note for note in summary['notes']))
            self.assertEqual(summary['display_shape'][0], self.default_summary_sample_size)
        else: # This case should not be hit given num_rows = 25000
            self.assertNotIn('notes', summary, "Notes should not be present if num_rows <= sample_size (but test expects sampling).")
            self.assertEqual(summary['display_shape'][0], num_rows)


    def test_upload_data_small_csv_no_sampling(self):
        """Test /upload-data with a small CSV that should not trigger sampling."""
        num_rows = self.default_summary_sample_size - 500
        if num_rows < 1: num_rows = 50
        num_cols = 3
        test_file_path = self._create_csv_file(num_rows, num_cols, "test_small.csv")

        with open(test_file_path, 'rb') as f:
            response = self.client.post("/upload-data", files={"file": ("test_small.csv", f, "text/csv")})

        self.assertEqual(response.status_code, 200, f"Response content: {response.content}")
        result = response.json()
        self.assertTrue(result['success'])
        summary = result['summary']

        self.assertNotIn('notes', summary, "Summary should not contain 'notes' for small CSVs.")
        self.assertEqual(summary['shape'][0], num_rows)
        self.assertEqual(summary['display_shape'][0], num_rows)

    def test_upload_data_empty_rows_csv(self):
        """Test /upload-data with a CSV file that has headers but no data rows."""
        empty_csv_path = "test_empty.csv"
        with open(empty_csv_path, 'w') as f:
            f.write("col1,col2,col3\n") # Header only, no data lines

        with open(empty_csv_path, 'rb') as f:
            response = self.client.post("/upload-data", files={"file": ("test_empty.csv", f, "text/csv")})

        self.assertEqual(response.status_code, 200, f"Response content: {response.content}")
        result = response.json()
        self.assertTrue(result['success'])
        summary = result['summary']
        self.assertEqual(summary['shape'][0], 0, "Shape should be 0 rows for empty CSV (header only).")


    def test_upload_data_completely_empty_csv_fails(self):
        """Test /upload-data with a completely empty CSV file (0 bytes)."""
        empty_file_path = "test_empty.csv"
        with open(empty_file_path, 'w') as f:
            pass # Create an empty file

        with open(empty_file_path, 'rb') as f:
            response = self.client.post("/upload-data", files={"file": ("test_empty.csv", f, "text/csv")})

        self.assertEqual(response.status_code, 400, f"Response content: {response.content}")
        result = response.json()
        self.assertIn("detail", result, "Error response should have details.")
        self.assertTrue("Error processing CSV file content" in result['detail'] or \
                        "Failed to read or parse CSV data" in result['detail'] or \
                        "No columns to parse from file" in result['detail'], # Pandas C error for empty
                        f"Unexpected error detail: {result['detail']}")


    def test_upload_data_invalid_file_type_not_csv(self):
        """Test /upload-data with a non-CSV file (e.g., a binary file)."""
        invalid_file_path = "test_invalid.bin" # Changed filename
        # Create a dummy binary file (e.g. some non-UTF8 bytes)
        # PNG file signature is a good example of non-text data.
        binary_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
        with open(invalid_file_path, 'wb') as f: # write in binary mode
            f.write(binary_content)

        with open(invalid_file_path, 'rb') as f:
            # Use a generic content type for binary data
            response = self.client.post("/upload-data", files={"file": (invalid_file_path, f, "application/octet-stream")})

        self.assertEqual(response.status_code, 400, f"Response content: {response.content}")
        result = response.json()
        self.assertIn("detail", result)
        # Check for a more specific error message related to CSV parsing or file content decoding
        self.assertTrue("Error processing CSV file content" in result['detail'] or \
                        "Error processing file: 'utf-8' codec can't decode byte" in result['detail'] or \
                        "Error processing file: Error tokenizing data" in result['detail'], # Pandas error for unparsable data
                        f"Unexpected error detail: {result['detail']}")

if __name__ == '__main__':
    # This allows running the tests directly using `python Hosting Web/test_app_new.py`
    unittest.main()
