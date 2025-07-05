import joblib
import geopandas as gpd
import pandas as pd
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List

# ==============================================================================
# --- APPLICATION SETUP ---
# ==============================================================================

# RUN THIS TO RUN THE BACKEND API
# cd backend
# uvicorn api:app --reload

# Create the FastAPI application instance
app = FastAPI(
    title="Urban Sentinel API",
    description="Predictive analytics for urban blight detection",
    version="1.0.0"
)

# --- CORS Middleware ---
# This allows your React frontend (running on a different address)
# to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],  # React dev server
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# --- Load Your Assets (Model and Data) ---
# This happens only ONCE when the API starts up.
def load_assets():
    """Load model and data assets with proper error handling."""
    try:
        # Try to load from current directory first, then from parent directory
        model_path = "urban_sentinel_model.pkl"
        geojson_path = "toronto_grid_with_temporal_features.geojson"
        
        if not os.path.exists(model_path):
            model_path = "../grid/urban_sentinel_model.pkl"
        if not os.path.exists(geojson_path):
            geojson_path = "../grid/toronto_grid_with_temporal_features.geojson"
            
        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        
        print(f"Loading data from: {geojson_path}")
        gdf = gpd.read_file(geojson_path)
        
        print(f"Assets loaded successfully! Data shape: {gdf.shape}")
        return model, gdf
    except Exception as e:
        print(f"Error loading assets: {e}")
        raise e

print("Loading model and data assets...")
model, gdf = load_assets()

# ==============================================================================
# --- API ENDPOINTS ---
# ==============================================================================

@app.get("/")
def read_root() -> Dict[str, str]:
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Welcome to the Urban Sentinel Predictive Blight API"}

@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": gdf is not None,
        "total_cells": len(gdf) if gdf is not None else 0
    }

@app.get("/api/predict-risk")
def predict_risk() -> List[Dict[str, Any]]:
    """
    This is the main endpoint for your application.
    It uses the trained model to predict a risk score for every grid cell
    and returns the full GeoJSON with this new information.
    """
    try:
        print("Received request to predict risk scores...")

        # Step 1: Prepare the feature data (X) exactly as you did for training
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        columns_to_drop = [
            'cell_id', 'is_blighted', 'target_blight_count',
            'overall_most_common_blight', 'recent_most_common_blight'
        ]
        
        # Filter out columns that don't exist
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        X_features = df.drop(columns=existing_columns)

        # Step 2: Use the model to predict probabilities (the "risk score")
        # model.predict_proba gives two columns: [prob_of_0, prob_of_1]
        # We want the probability of being blighted (class 1).
        risk_scores = model.predict_proba(X_features)[:, 1]
        
        # Step 3: Add this new risk score to our original GeoDataFrame
        gdf_copy = gdf.copy()
        gdf_copy['risk_score'] = risk_scores
        print(f"  - Calculated risk scores for {len(gdf_copy)} cells.")

        # Step 4: Return the entire GeoJSON.
        # FastAPI will automatically convert this to a JSON response.
        return gdf_copy.to_dict(orient='records')
        
    except Exception as e:
        print(f"Error in predict_risk: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/stats")
def get_statistics() -> Dict[str, Any]:
    """Get basic statistics about the data."""
    try:
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        
        stats = {
            "total_cells": len(df),
            "columns": list(df.columns),
        }
        
        # Add statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            stats[f"{col}_mean"] = float(df[col].mean())
            stats[f"{col}_std"] = float(df[col].std())
            
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics calculation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)