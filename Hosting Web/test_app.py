#!/usr/bin/env python3
"""
Test script to verify the Propensity Matching Tool works correctly
"""

import pandas as pd
import json
import time
import requests
from pathlib import Path

def test_data_processing():
    """Test the data processing pipeline"""
    print("🧪 Testing data processing...")
    
    try:
        from core.data_processor import DataProcessor
        from core.visualizations import MatchingVisualizer
        
        # Create test data
        test_data = {
            'org_id': [1, 2, 3, 4, 5],
            'region': ['US', 'EU', 'US', 'EU', 'US'],
            'mrr': [1000, 1500, 800, 1200, 900],
            'treatment_flag': [1, 0, 1, 0, 1],
            'ds_month': ['2024-01'] * 5
        }
        df = pd.DataFrame(test_data)
        
        # Test data processor
        processor = DataProcessor()
        summary = processor.get_data_summary(df)
        
        # Test JSON serialization
        json_str = json.dumps(summary)
        
        print("✅ Data processing test passed!")
        print(f"   - Shape: {summary['shape']}")
        print(f"   - Columns: {len(summary['columns'])}")
        print(f"   - JSON serializable: ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False

def test_server_health():
    """Test if the server is running"""
    print("🌐 Testing server health...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Server health test passed!")
            print(f"   - Status: {data.get('status')}")
            print(f"   - Version: {data.get('version')}")
            return True
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Server not accessible: {e}")
        print("   💡 Make sure to start the server first:")
        print("      python3 app.py")
        return False

def test_file_upload():
    """Test file upload functionality"""
    print("📁 Testing file upload...")
    
    # Check if test file exists
    test_file = Path("test_data.csv")
    if not test_file.exists():
        print("❌ Test file not found. Creating one...")
        create_test_file()
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': ('test_data.csv', f, 'text/csv')}
            response = requests.post("http://localhost:8000/upload-data", files=files, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ File upload test passed!")
                summary = data.get('summary', {})
                print(f"   - Uploaded shape: {summary.get('shape')}")
                print(f"   - Columns: {len(summary.get('columns', []))}")
                return True
            else:
                print(f"❌ Upload failed: {data}")
                return False
        else:
            print(f"❌ Upload request failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Upload request failed: {e}")
        return False

def create_test_file():
    """Create a test CSV file"""
    test_data = {
        'org_id': range(1, 21),
        'region': ['US', 'EU'] * 10,
        'channel_group': ['Online', 'Sales'] * 10,
        'mrr': [1000 + i*50 for i in range(20)],
        'treatment_flag': [1, 0] * 10,
        'ds_month': ['2024-01'] * 20
    }
    
    df = pd.DataFrame(test_data)
    df.to_csv('test_data.csv', index=False)
    print("✅ Test file created: test_data.csv")

def main():
    """Run all tests"""
    print("🧑‍💻 Propensity Matching Tool - Test Suite")
    print("=" * 50)
    
    tests = [
        test_data_processing,
        test_server_health,
        test_file_upload
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    print("📊 Test Results Summary")
    print("-" * 30)
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your application is ready to use.")
        print("\n📖 Next steps:")
        print("1. Open http://localhost:8000 in your browser")
        print("2. Upload your CSV file")
        print("3. Configure matching parameters")
        print("4. Run propensity matching")
        print("5. Download results and view visualizations")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        
        if not results[1]:  # Server health failed
            print("\n🚀 To start the server:")
            print("   python3 app.py")
            print("   # OR")
            print("   python3 start.py")

if __name__ == "__main__":
    main() 