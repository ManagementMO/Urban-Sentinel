#!/usr/bin/env python3
"""
Test script for geocode_wards_csv.py
Validates the implementation and checks for runtime issues.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

def create_test_data():
    """Create minimal test data for validation."""
    print("🧪 Creating test data...")
    
    # Create sample service requests data
    np.random.seed(42)
    
    # Generate dates from 2014 to 2024
    start_date = datetime(2014, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Generate 1000 sample records
    n_records = 1000
    
    # Random dates
    date_range = (end_date - start_date).days
    random_days = np.random.randint(0, date_range, n_records)
    dates = [start_date + timedelta(days=int(day)) for day in random_days]
    
    # Sample wards
    wards = ['Ward 1', 'Ward 2', 'Ward 3', 'Ward 4', 'Ward 5']
    
    # Sample service types (including blight indicators)
    service_types = [
        'Road - Pot hole',
        'Graffiti',
        'Property Standards',
        'Garbage Collection - Missed Pick-Up',
        'Noise Complaint',
        'Sidewalk - Damaged / Concrete',
        'Long Grass and Weeds',
        'Litter / Bin / Overflow or Not Picked Up',
        'Road - Damaged',
        'Other Service Request'
    ]
    
    # Create test dataframe
    test_data = pd.DataFrame({
        'Creation Date': dates,
        'Ward': np.random.choice(wards, n_records),
        'Service Request Type': np.random.choice(service_types, n_records),
        'Division': np.random.choice(['Division A', 'Division B', 'Division C'], n_records),
        'Status': np.random.choice(['Open', 'Closed', 'In Progress'], n_records)
    })
    
    # Save test data
    test_data.to_csv('test_service_requests.csv', index=False)
    print(f"   ✓ Created {len(test_data)} test records")
    print(f"   ✓ Date range: {test_data['Creation Date'].min()} to {test_data['Creation Date'].max()}")
    print(f"   ✓ Unique wards: {test_data['Ward'].nunique()}")
    print(f"   ✓ Unique service types: {test_data['Service Request Type'].nunique()}")
    
    return 'test_service_requests.csv'

def test_geocode_wards_csv():
    """Test the geocode_wards_csv implementation."""
    print("\n🧪 Testing geocode_wards_csv.py...")
    
    try:
        # Import the module
        sys.path.append('data_preparation')
        from geocode_wards_csv import WardBasedModelDataGeneratorCSV
        
        # Create test configuration
        config = {
            'service_requests_csv': 'test_service_requests.csv',
            'ward_boundaries_file': 'nonexistent_ward_data.geojson',  # Test without boundaries
            'output_csv': 'test_model_ready_data.csv',
            'history_start_year': 2014,
            'history_end_year': 2023,
            'target_year': 2024
        }
        
        # Test the generator
        generator = WardBasedModelDataGeneratorCSV(config)
        
        # Test each step
        print("\n1. Testing data loading...")
        if not generator.load_and_validate_data():
            print("❌ Data loading failed")
            return False
        print("✅ Data loading successful")
        
        print("\n2. Testing data cleaning...")
        if not generator.clean_and_prepare_data():
            print("❌ Data cleaning failed")
            return False
        print("✅ Data cleaning successful")
        
        print("\n3. Testing feature creation...")
        if not generator.create_advanced_features():
            print("❌ Feature creation failed")
            return False
        print("✅ Feature creation successful")
        
        print("\n4. Testing finalization...")
        if not generator.merge_and_finalize():
            print("❌ Finalization failed")
            return False
        print("✅ Finalization successful")
        
        print("\n5. Testing save...")
        if not generator.save_results():
            print("❌ Save failed")
            return False
        print("✅ Save successful")
        
        # Check output
        if os.path.exists('test_model_ready_data.csv'):
            df = pd.read_csv('test_model_ready_data.csv')
            print(f"\n📊 Output Summary:")
            print(f"   • Rows: {len(df)}")
            print(f"   • Columns: {len(df.columns)}")
            print(f"   • Features: {len([col for col in df.columns if not col.startswith('ward_name')])}")
            
            # Check for required columns
            required_cols = ['is_high_blight_risk', 'is_extreme_blight_risk', 'risk_level', 'risk_score']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"   ⚠️  Missing target columns: {missing_cols}")
            else:
                print("   ✅ All target columns present")
                
            # Check for missing values
            missing_values = df.isnull().sum().sum()
            if missing_values > 0:
                print(f"   ⚠️  Missing values: {missing_values}")
            else:
                print("   ✅ No missing values")
                
            # Check data types
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            print(f"   ✅ Numeric columns: {len(numeric_cols)}")
            
            return True
        else:
            print("❌ Output file not created")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup():
    """Clean up test files."""
    test_files = [
        'test_service_requests.csv',
        'test_model_ready_data.csv',
        'test_model_ready_data_metadata.json'
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"   🗑️  Removed {file}")

if __name__ == "__main__":
    print("🧪 GEOCODE WARDS CSV VALIDATION TEST")
    print("=" * 50)
    
    try:
        # Create test data
        test_file = create_test_data()
        
        # Run tests
        success = test_geocode_wards_csv()
        
        if success:
            print("\n✅ ALL TESTS PASSED!")
            print("   The geocode_wards_csv.py implementation is working correctly.")
            print("   Ready for gradient boosting model training!")
        else:
            print("\n❌ TESTS FAILED!")
            print("   The implementation needs fixes before use.")
            
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up test files...")
        cleanup()
        
    sys.exit(0 if success else 1)