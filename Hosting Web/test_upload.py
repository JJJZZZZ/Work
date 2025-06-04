#!/usr/bin/env python3
"""
Simple test script to test file upload functionality
"""

import requests
import time

def test_upload():
    url = "http://localhost:8000/upload-data"
    
    try:
        print("Testing file upload...")
        
        # Read the test file
        with open('test_data.csv', 'rb') as f:
            files = {'file': ('test_data.csv', f, 'text/csv')}
            
            print("Sending request...")
            start_time = time.time()
            
            response = requests.post(url, files=files, timeout=30)
            
            end_time = time.time()
            print(f"Request completed in {end_time - start_time:.2f} seconds")
            
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text[:500]}...")  # First 500 chars
            
            if response.status_code == 200:
                print("✅ Upload successful!")
            else:
                print("❌ Upload failed!")
                
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 30 seconds")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_upload()
