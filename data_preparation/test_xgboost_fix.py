#!/usr/bin/env python3
"""
Quick test to verify XGBoost API fix works correctly.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import xgboost as xgb

def test_xgboost_api():
    """Test that XGBoost API works correctly with current version."""
    print("🧪 Testing XGBoost API compatibility...")
    
    # Create sample data
    X, y = make_classification(
        n_samples=100, 
        n_features=10, 
        n_classes=2, 
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Test parameters that should work
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': 42,
        'verbosity': 0
    }
    
    try:
        # Test model creation
        model = xgb.XGBClassifier(**params)
        print("   ✓ XGBClassifier created successfully")
        
        # Test fitting with eval_set
        eval_set = [(X_train, y_train), (X_test, y_test)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        print("   ✓ Model fitted successfully")
        
        # Test predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        print("   ✓ Predictions generated successfully")
        
        # Test feature importance
        importance = model.feature_importances_
        print("   ✓ Feature importance extracted successfully")
        
        # Test attributes
        if hasattr(model, 'best_iteration'):
            print(f"   ✓ Best iteration: {model.best_iteration}")
        else:
            print(f"   ✓ Used {params['n_estimators']} estimators")
            
        print(f"\n🎉 XGBoost API test passed!")
        print(f"   • XGBoost version: {xgb.__version__}")
        print(f"   • Predictions shape: {predictions.shape}")
        print(f"   • Probabilities shape: {probabilities.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ XGBoost API test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_xgboost_api()
    if success:
        print("\n✅ The XGBoost training script should now work correctly!")
    else:
        print("\n❌ There may still be XGBoost API issues to resolve.") 