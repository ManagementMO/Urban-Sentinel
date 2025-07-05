import geopandas as gpd
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# The name of the feature-rich GeoJSON file you just created.
INPUT_GEOJSON = "toronto_grid_with_temporal_features.geojson"

# The name of the file where the trained model will be saved.
OUTPUT_MODEL_FILE = "urban_sentinel_model.pkl"

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
        'target_blight_count' # This column was used to create the target, so it MUST be removed.
    ]
    X = df.drop(columns=columns_to_drop)

    print(f"  - Using {len(X.columns)} features to predict the target '{TARGET_COLUMN}'.")
    print(f"  - Number of blighted cells (target=1): {y.sum()}")
    print(f"  - Number of non-blighted cells (target=0): {len(y) - y.sum()}")


    # --- Step 3: Choose and Train the Model ---
    print("\n--- Step 3: Training the LightGBM classification model... ---")

    # We use a LightGBM Classifier, which is a powerful and fast algorithm for this type of problem.
    # 'objective='binary'' tells the model this is a yes/no prediction.
    # 'random_state=42' ensures we get the same result every time we run the script.
    model = lgb.LGBMClassifier(objective='binary', random_state=42)

    # This is the line where the model "learns" the patterns from your data.
    model.fit(X, y)
    print("  - Model training complete.")


    # --- Step 4: Evaluate Model Performance ---
    print("\n--- Step 4: Evaluating model performance using cross-validation... ---")

    # Cross-validation is a robust method to estimate how the model will perform on new, unseen data.
    # It splits the data into 5 parts (cv=5), trains on 4, and tests on 1, rotating through all parts.
    print("  - Calculating cross-validated scores (this may take a moment)...")
    accuracy_cv = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    precision_cv = cross_val_score(model, X, y, cv=5, scoring='precision').mean()
    recall_cv = cross_val_score(model, X, y, cv=5, scoring='recall').mean()
    f1_cv = cross_val_score(model, X, y, cv=5, scoring='f1').mean()

    print("\n--- Model Performance Metrics ---")
    print(f"  Accuracy:  {accuracy_cv:.2%}")
    print(f"  Precision: {precision_cv:.2%}")
    print(f"  Recall:    {recall_cv:.2%}")
    print(f"  F1-Score:  {f1_cv:.2%}")
    print("---------------------------------")
    print("  - Precision: Of all the cells our model predicts as 'blighted', how many actually were?")
    print("  - Recall: Of all the cells that were truly 'blighted', how many did our model successfully find?")


    # --- Step 5: Analyze Feature Importance ---
    print("\n--- Step 5: Analyzing which features are most important... ---")

    # This tells us which of our engineered features the model found most predictive.
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("  - Top 15 Most Important Features:")
    print(feature_importances.head(15).to_string(index=False))


    # --- Step 6: Save the Trained Model for Later Use ---
    print(f"\n--- Step 6: Saving the trained model to '{OUTPUT_MODEL_FILE}'... ---")

    # We save the 'model' object, which contains all of its learned knowledge.
    # This file is the "brain" that we will load into our backend API.
    joblib.dump(model, OUTPUT_MODEL_FILE)
    print("  - Model saved successfully.")
    print("\n==================== PROCESS COMPLETE ====================")
    print("You are now ready to build the backend API to serve your model's predictions.")


if __name__ == '__main__':
    train_and_evaluate_model()