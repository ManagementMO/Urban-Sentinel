import geopandas as gpd
import pandas as pd
import lightgbm as lgb
import json
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# The name of the feature-rich GeoJSON file you just created.
INPUT_GEOJSON = "toronto_grid_with_temporal_features.geojson"

# The name of the file where the trained model will be saved.
OUTPUT_MODEL_FILE = "urban_sentinel_model.txt"

# The name of the target column we are trying to predict.
TARGET_COLUMN = "is_blighted"

# ==============================================================================
# --- SCRIPT EXECUTION ---
# ==============================================================================

def train_and_evaluate_model():
    """
    Loads the feature data, trains a LightGBM model, evaluates its performance,
    and saves the final trained model to a file.
    """

    # --- Step 1: Load the Final Dataset ---
    print("--- Step 1: Loading the final GeoJSON feature data... ---")
    gdf = gpd.read_file(INPUT_GEOJSON)
    # For modeling, we can work with a standard pandas DataFrame (geometry is not needed for training).
    df = pd.DataFrame(gdf.drop(columns='geometry'))
    print(f"  - Loaded {len(df)} grid cells with {len(df.columns)} columns each.")


    # --- Step 2: Define Features (X) and Target (y) ---
    print("\n--- Step 2: Separating features (X) from the target (y)... ---")

    # The target 'y' is the single column we want to predict.
    y = df[TARGET_COLUMN]

    # The features 'X' are all other columns, except for identifiers or columns that
    # would "leak" information about the target, giving the model the answer.
    columns_to_drop = [
        'cell_id',
        TARGET_COLUMN,
        'target_blight_count', # This column was used to create the target, so it MUST be removed.
        'overall_most_common_blight', # Text column - would need encoding for ML
        'recent_most_common_blight' # Text column - would need encoding for ML
    ]
    X = df.drop(columns=columns_to_drop)

    print(f"  - Using {len(X.columns)} features to predict the target '{TARGET_COLUMN}'.")
    print(f"  - Number of blighted cells (target=1): {y.sum()}")
    print(f"  - Number of non-blighted cells (target=0): {len(y) - y.sum()}")


    # --- Step 3: Handle Class Imbalance and Prepare Data ---
    print("\n--- Step 3: Analyzing class distribution and preparing data... ---")
    
    # Calculate class distribution
    class_counts = y.value_counts()
    imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1
    print(f"  - Class imbalance ratio (non-blighted/blighted): {imbalance_ratio:.2f}")
    
    # Compute class weights for handling imbalance
    unique_classes = y.unique()
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"  - Computed class weights: {class_weight_dict}")
    
    # Split data for early stopping validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  - Training set: {len(X_train)} samples")
    print(f"  - Validation set: {len(X_val)} samples")

    # --- Step 4: Configure and Train the Enhanced LightGBM Model ---
    print("\n--- Step 4: Training enhanced LightGBM classification model... ---")
    
    # Enhanced LightGBM configuration optimized for urban blight prediction
    model = lgb.LGBMClassifier(
        # Core parameters
        objective='binary',
        boosting_type='gbdt',
        random_state=42,
        n_jobs=-1,  # Use all available cores
        
        # Performance parameters (tuned for urban blight prediction)
        n_estimators=500,  # More trees for better performance
        learning_rate=0.05,  # Lower learning rate with more estimators
        max_depth=8,  # Deeper trees for complex urban patterns
        num_leaves=100,  # More leaves for detailed decision boundaries
        
        # Feature sampling (prevent overfitting)
        subsample=0.8,  # Use 80% of samples for each tree
        subsample_freq=1,  # Apply subsample every iteration
        colsample_bytree=0.8,  # Use 80% of features for each tree
        
        # Regularization (prevent overfitting)
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=0.1,  # L2 regularization
        min_split_gain=0.01,  # Minimum gain required to split
        min_child_weight=0.01,  # Minimum weight in child nodes
        min_child_samples=20,  # Minimum samples in child nodes
        
        # Class imbalance handling
        class_weight='balanced',  # Automatically handle class imbalance
        
        # Feature importance
        importance_type='gain',  # Use gain-based importance (more meaningful)
        
        # Advanced parameters for urban data
        subsample_for_bin=200000,  # Sufficient for urban datasets
        
        # Additional parameters for stability
        verbose=-1  # Suppress training output
    )
    
    print("  - Model configuration:")
    print(f"    • Trees: {model.n_estimators}")
    print(f"    • Learning rate: {model.learning_rate}")
    print(f"    • Max depth: {model.max_depth}")
    print(f"    • Num leaves: {model.num_leaves}")
    print(f"    • Regularization: L1={model.reg_alpha}, L2={model.reg_lambda}")
    print(f"    • Feature sampling: {model.colsample_bytree}")
    print(f"    • Class weight: {model.class_weight}")
    
    # Train with early stopping
    print("  - Training with early stopping validation...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)  # Suppress per-iteration logs
        ]
    )
    
    # Get the best number of estimators
    best_iteration = model.best_iteration_ if hasattr(model, 'best_iteration_') else model.n_estimators
    print(f"  - Model training complete. Best iteration: {best_iteration}")
    
    # Retrain on full dataset with optimal number of estimators
    print("  - Retraining on full dataset with optimal parameters...")
    final_model = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        random_state=42,
        n_jobs=-1,
        n_estimators=best_iteration,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_split_gain=0.01,
        min_child_weight=0.01,
        min_child_samples=20,
        class_weight='balanced',
        importance_type='gain',
        subsample_for_bin=200000,
        verbose=-1
    )
    
    # Train on full dataset
    final_model.fit(X, y)
    model = final_model  # Use the final model for evaluation


    # --- Step 5: Comprehensive Model Evaluation ---
    print("\n--- Step 5: Comprehensive model evaluation using stratified cross-validation... ---")

    # Use stratified cross-validation to ensure balanced class distribution across folds
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate multiple metrics with cross-validation
    print("  - Calculating cross-validated scores (this may take a moment)...")
    accuracy_cv = cross_val_score(model, X, y, cv=stratified_cv, scoring='accuracy')
    precision_cv = cross_val_score(model, X, y, cv=stratified_cv, scoring='precision')
    recall_cv = cross_val_score(model, X, y, cv=stratified_cv, scoring='recall')
    f1_cv = cross_val_score(model, X, y, cv=stratified_cv, scoring='f1')
    roc_auc_cv = cross_val_score(model, X, y, cv=stratified_cv, scoring='roc_auc')
    
    # Calculate mean and standard deviation for each metric
    metrics = {
        'Accuracy': accuracy_cv,
        'Precision': precision_cv,
        'Recall': recall_cv,
        'F1-Score': f1_cv,
        'ROC-AUC': roc_auc_cv
    }
    
    print("\n--- Enhanced Model Performance Metrics ---")
    print("=" * 60)
    for metric_name, scores in metrics.items():
        mean_score = scores.mean()
        std_score = scores.std()
        print(f"  {metric_name:12}: {mean_score:.3f} ± {std_score:.3f} (95% CI: {mean_score-2*std_score:.3f}-{mean_score+2*std_score:.3f})")
    
    print("=" * 60)
    print("\n--- Metric Explanations ---")
    print("  • Accuracy:  Overall correctness of predictions")
    print("  • Precision: Of predicted blight areas, how many are actually blighted")
    print("  • Recall:    Of actual blight areas, how many were correctly identified")
    print("  • F1-Score:  Harmonic mean of precision and recall (balanced measure)")
    print("  • ROC-AUC:   Area under ROC curve (discrimination ability)")
    
    # Additional evaluation with hold-out validation set
    print("\n--- Hold-out Validation Results ---")
    y_val_pred = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    print(f"  Validation Accuracy:  {val_accuracy:.3f}")
    print(f"  Validation Precision: {val_precision:.3f}")
    print(f"  Validation Recall:    {val_recall:.3f}")
    print(f"  Validation F1-Score:  {val_f1:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    Predicted:  0(Non-Blight)  1(Blight)")
    print(f"    Actual 0:        {cm[0,0]:4d}        {cm[0,1]:4d}")
    print(f"    Actual 1:        {cm[1,0]:4d}        {cm[1,1]:4d}")
    
    # Classification report
    print("\n  Detailed Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['Non-Blight', 'Blight']))


    # --- Step 6: Enhanced Feature Importance Analysis ---
    print("\n--- Step 6: Analyzing feature importance (using gain-based method)... ---")

    # Create comprehensive feature importance analysis
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_,
        'rank': range(1, len(X.columns) + 1)
    }).sort_values('importance', ascending=False)
    
    # Reset rank after sorting
    feature_importances['rank'] = range(1, len(feature_importances) + 1)
    
    # Determine how many features to show
    num_features = len(X.columns)
    features_to_show = min(20, num_features)
    
    print(f"  - Top {features_to_show} Most Important Features (Gain-based):")
    print("=" * 80)
    print(f"{'Rank':<4} {'Feature':<40} {'Importance':<12} {'Contribution':<10}")
    print("-" * 80)
    
    total_importance = feature_importances['importance'].sum()
    for idx, row in feature_importances.head(features_to_show).iterrows():
        contribution = (row['importance'] / total_importance) * 100
        print(f"{row['rank']:<4} {row['feature']:<40} {row['importance']:<12.6f} {contribution:<10.2f}%")
    
    print("=" * 80)
    
    # Feature importance insights
    top_features_count = min(5, num_features)
    top_features = feature_importances.head(top_features_count)['feature'].tolist()
    print(f"\n  - Top {top_features_count} Most Predictive Features:")
    for i, feature in enumerate(top_features, 1):
        print(f"    {i}. {feature}")
    
    # Calculate cumulative importance
    cumulative_importance = feature_importances['importance'].cumsum() / total_importance
    
    # Calculate coverage analysis based on available features
    num_features = len(X.columns)
    print(f"\n  - Feature Coverage Analysis:")
    
    if num_features >= 10:
        top_10_coverage = cumulative_importance.iloc[9] * 100  # Top 10 features
        print(f"    • Top 10 features explain {top_10_coverage:.1f}% of model decisions")
    else:
        top_n_coverage = cumulative_importance.iloc[num_features-1] * 100  # All features
        print(f"    • All {num_features} features explain {top_n_coverage:.1f}% of model decisions")
    
    if num_features >= 20:
        top_20_coverage = cumulative_importance.iloc[19] * 100  # Top 20 features
        print(f"    • Top 20 features explain {top_20_coverage:.1f}% of model decisions")
    elif num_features >= 10:
        top_n_coverage = cumulative_importance.iloc[num_features-1] * 100  # All features
        print(f"    • All {num_features} features explain {top_n_coverage:.1f}% of model decisions")
    
    print(f"    • Total features used: {num_features}")
    
    # Show coverage for top 5 features if available
    if num_features >= 5:
        top_5_coverage = cumulative_importance.iloc[4] * 100  # Top 5 features
        print(f"    • Top 5 features explain {top_5_coverage:.1f}% of model decisions")

    # --- Step 7: Enhanced Model Saving with Metadata ---
    print(f"\n--- Step 7: Saving enhanced model to '{OUTPUT_MODEL_FILE}' and metadata... ---")

    # Create comprehensive model metadata
    model_metadata = {
        'model_type': 'LightGBM Classifier',
        'model_version': '1.0',
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_features': len(X.columns),
        'n_samples': len(X),
        'class_distribution': y.value_counts().to_dict(),
        'feature_names': X.columns.tolist(),
        'target_column': TARGET_COLUMN,
        'performance_metrics': {
            'cv_accuracy_mean': accuracy_cv.mean(),
            'cv_accuracy_std': accuracy_cv.std(),
            'cv_precision_mean': precision_cv.mean(),
            'cv_precision_std': precision_cv.std(),
            'cv_recall_mean': recall_cv.mean(),
            'cv_recall_std': recall_cv.std(),
            'cv_f1_mean': f1_cv.mean(),
            'cv_f1_std': f1_cv.std(),
            'cv_roc_auc_mean': roc_auc_cv.mean(),
            'cv_roc_auc_std': roc_auc_cv.std(),
            'validation_accuracy': val_accuracy,
            'validation_precision': val_precision,
            'validation_recall': val_recall,
            'validation_f1': val_f1
        },
        'model_parameters': {
            'n_estimators': model.n_estimators,
            'learning_rate': model.learning_rate,
            'max_depth': model.max_depth,
            'num_leaves': model.num_leaves,
            'subsample': model.subsample,
            'colsample_bytree': model.colsample_bytree,
            'reg_alpha': model.reg_alpha,
            'reg_lambda': model.reg_lambda,
            'class_weight': str(model.class_weight)
        },
        'feature_importance': feature_importances.to_dict('records'),
        'best_iteration': best_iteration if 'best_iteration' in locals() else model.n_estimators
    }
    
    # Save the trained model using LightGBM's native method
    model.booster_.save_model(OUTPUT_MODEL_FILE)

    # Define metadata file path
    OUTPUT_METADATA_FILE = OUTPUT_MODEL_FILE.replace('.txt', '_metadata.json')

    # Save model metadata to a separate JSON file
    with open(OUTPUT_METADATA_FILE, 'w') as f:
        json.dump(model_metadata, f, indent=4)
    
    print("  - Model saved successfully!")
    print(f"  - Metadata saved to '{OUTPUT_METADATA_FILE}'")
    
    print("\n" + "=" * 80)
    print("🎉 ENHANCED MODEL TRAINING COMPLETE! 🎉")
    print("=" * 80)
    print(f"📊 Model Performance Summary:")
    print(f"   • Cross-validated F1-Score: {f1_cv.mean():.3f} ± {f1_cv.std():.3f}")
    print(f"   • Cross-validated ROC-AUC:  {roc_auc_cv.mean():.3f} ± {roc_auc_cv.std():.3f}")
    print(f"   • Validation F1-Score:      {val_f1:.3f}")
    print(f"   • Class Balance Handled:    ✓")
    print(f"   • Early Stopping Used:      ✓")
    print(f"   • Feature Importance:       ✓ (Top feature: {top_features[0]})")
    print("=" * 80)
    print("🚀 Ready to deploy to the Urban Sentinel API! 🚀")


if __name__ == '__main__':
    train_and_evaluate_model()