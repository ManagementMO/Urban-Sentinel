import lightgbm as lgb
import json
import geopandas as gpd
import pandas as pd
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Union

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
# Configure CORS for production deployment
cors_origins = [
    "http://localhost:3000",  # Local development
    "http://127.0.0.1:3000",  # Local development
    "https://*.onrender.com",  # Render deployment
    os.getenv("FRONTEND_URL", "https://urban-sentinel-frontend.onrender.com")  # Production frontend
]

# When running in a local Docker container, the origin might be unpredictable.
# Using a wildcard for development is a common and safe practice.
if os.getenv("ENVIRONMENT", "development").lower() == "development":
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Your Assets (Model and Data) ---
# This happens only ONCE when the API starts up.
def load_assets():
    """Load model and data assets with proper error handling."""
    try:
        # Define paths for model, metadata, and GeoJSON files
        model_path = "urban_sentinel_model.txt"
        metadata_path = "urban_sentinel_model_metadata.json"
        geojson_path = "toronto_grid_with_temporal_features.geojson"

        # Check for files in parent directory if not in current (for Docker context)
        if not os.path.exists(model_path):
            model_path = os.path.join("..", "grid", model_path)
        if not os.path.exists(metadata_path):
            metadata_path = os.path.join("..", "grid", metadata_path)
        if not os.path.exists(geojson_path):
            geojson_path = os.path.join("..", "grid", geojson_path)

        print(f"Loading model from: {model_path}")
        model = lgb.Booster(model_file=model_path)
        print("✓ Model loaded successfully (LightGBM native).")

        print(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, 'r') as f:
            model_metadata = json.load(f)

        # Store metadata globally and convert feature importances to DataFrame
        global model_metadata_global, feature_importances_global
        model_metadata_global = model_metadata
        if 'feature_importance' in model_metadata:
            feature_importances_global = pd.DataFrame(model_metadata['feature_importance'])
        else:
            feature_importances_global = None
        
        print("✓ Metadata loaded successfully!")
        print(f"  - Model type: {model_metadata.get('model_type', 'Unknown')}")
        print(f"  - Model version: {model_metadata.get('model_version', 'Unknown')}")
        print(f"  - Training date: {model_metadata.get('training_date', 'Unknown')}")
        
        print(f"Loading data from: {geojson_path}")
        gdf = gpd.read_file(geojson_path)
        
        print(f"Assets loaded successfully! Data shape: {gdf.shape}")
        return model, gdf
    except Exception as e:
        print(f"Error loading assets: {e}")
        raise e

# Global variables for model metadata
model_metadata_global = {}
feature_importances_global = None

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
    """Enhanced health check endpoint with model metadata."""
    health_info = {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": gdf is not None,
        "total_cells": len(gdf) if gdf is not None else 0
    }
    
    # Add enhanced model metadata if available
    if model_metadata_global:
        health_info.update({
            "model_type": model_metadata_global.get("model_type", "Unknown"),
            "model_version": model_metadata_global.get("model_version", "Unknown"),
            "training_date": model_metadata_global.get("training_date", "Unknown"),
            "n_features": model_metadata_global.get("n_features", "Unknown"),
            "n_samples": model_metadata_global.get("n_samples", "Unknown"),
            "performance_metrics": model_metadata_global.get("performance_metrics", {}),
            "enhanced_model": True
        })
    else:
        health_info["enhanced_model"] = False
    
    return health_info

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
        # A LightGBM Booster's predict method returns probabilities for binary classification.
        risk_scores = model.predict(X_features)
        
        # Step 3: Add this new risk score to our original GeoDataFrame
        gdf_copy = gdf.copy()
        gdf_copy['risk_score'] = risk_scores
        print(f"  - Calculated risk scores for {len(gdf_copy)} cells.")

        # Step 4: Convert geometry to string format and return as JSON.
        # Convert geometry to WKT string for JSON serialization
        gdf_copy['geometry'] = gdf_copy['geometry'].apply(lambda x: x.wkt if x is not None else None)
        
        # FastAPI will automatically convert this to a JSON response.
        return gdf_copy.to_dict(orient='records')
        
    except Exception as e:
        print(f"Error in predict_risk: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/stats")
def get_statistics() -> Dict[str, Union[int, float, List[str]]]:
    """Get basic statistics about the data."""
    try:
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        
        stats: Dict[str, Union[int, float, List[str]]] = {
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

@app.get("/api/feature-importance")
def get_feature_importance() -> Dict[str, Any]:
    """
    Get enhanced feature importance from the trained model to understand what drives blight risk.
    Returns detailed feature importance analysis with gain-based rankings.
    """
    try:
        # Use enhanced feature importances if available
        if feature_importances_global is not None:
            # Enhanced feature importance from model package
            print("Using enhanced feature importance data...")
            
            top_features = []
            all_features = []
            
            for idx, row in feature_importances_global.iterrows():
                feature_data = {
                    "feature": row['feature'],
                    "importance": float(row['importance']),
                    "rank": int(row['rank']),
                    "description": get_feature_description(row['feature'])
                }
                
                all_features.append(feature_data)
                if row['rank'] <= 15:  # Top 15 features
                    top_features.append(feature_data)
            
            # Calculate total importance and contribution percentages
            total_importance = sum(row['importance'] for row in all_features)
            for feature_data in all_features:
                feature_data['contribution_percent'] = (feature_data['importance'] / total_importance) * 100
            
            return {
                "top_features": top_features[:15],
                "all_features": all_features,
                "total_features": len(all_features),
                "model_type": model_metadata_global.get('model_type', type(model).__name__),
                "importance_type": "gain",
                "enhanced": True,
                "total_importance": total_importance,
                "feature_coverage": {
                    "top_5": sum(f['contribution_percent'] for f in all_features[:5]),
                    "top_10": sum(f['contribution_percent'] for f in all_features[:10]),
                    "top_15": sum(f['contribution_percent'] for f in all_features[:15])
                }
            }
        
        else:
            # Fallback to legacy feature importance calculation
            print("Using legacy feature importance calculation...")
            
        # Get the feature names (same columns used for training)
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        columns_to_drop = [
            'cell_id', 'is_blighted', 'target_blight_count',
            'overall_most_common_blight', 'recent_most_common_blight'
        ]
        
        # Filter out columns that don't exist
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        feature_df = df.drop(columns=existing_columns)
        feature_names = list(feature_df.columns)
        
        # Get feature importances from the model
        if hasattr(model, 'feature_importance'):
            importances = model.feature_importance(importance_type='split')
        else:
            raise HTTPException(status_code=500, detail="Model does not support feature importance")
        
        # Create feature importance pairs and sort by importance
        feature_importance_pairs = list(zip(feature_names, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Format the response
        top_features = []
        all_features = []
        
        for i, (feature, importance) in enumerate(feature_importance_pairs):
            feature_data = {
                "feature": feature,
                "importance": float(importance),
                "rank": i + 1,
                "description": get_feature_description(feature)
            }
            
            all_features.append(feature_data)
            if i < 10:  # Top 10 features
                top_features.append(feature_data)
        
        return {
            "top_features": top_features,
            "all_features": all_features,
            "total_features": len(feature_names),
                "model_type": type(model).__name__,
                "importance_type": "split",
                "enhanced": False
        }
        
    except Exception as e:
        print(f"Error in get_feature_importance: {e}")
        raise HTTPException(status_code=500, detail=f"Feature importance calculation failed: {str(e)}")

def get_feature_description(feature_name: str) -> str:
    """Get human-readable description for feature names."""
    descriptions = {
        'total_complaints_mean': 'Average number of all 311 complaints per year',
        'total_complaints_std': 'Variability in total complaints over time',
        'blight_complaints_mean': 'Average number of blight-related complaints per year',
        'blight_complaints_std': 'Variability in blight complaints over time',
        'blight_complaints_recent_mean': 'Average blight complaints in recent years (2018+)',
        'blight_complaints_trend': 'Trend in blight complaints over time (increasing/decreasing)',
        'blight_complaints_2019': 'Number of blight complaints in 2019',
        'blight_complaints_2020': 'Number of blight complaints in 2020'
    }
    return descriptions.get(feature_name, f"Feature: {feature_name}")

@app.get("/api/cell-details/{cell_id}")
def get_cell_details(cell_id: int) -> Dict[str, Any]:
    """
    Get detailed information about a specific cell including its risk factors.
    """
    try:
        # Find the cell
        cell_data = gdf[gdf['cell_id'] == cell_id]
        if len(cell_data) == 0:
            raise HTTPException(status_code=404, detail=f"Cell {cell_id} not found")
        
        cell = cell_data.iloc[0]
        
        # Get feature values for this cell
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        columns_to_drop = [
            'cell_id', 'is_blighted', 'target_blight_count',
            'overall_most_common_blight', 'recent_most_common_blight'
        ]
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        feature_df = df.drop(columns=existing_columns)
        
        cell_features = feature_df[feature_df.index == cell_data.index[0]]
        
        # Get prediction for this cell
        risk_score = float(model.predict(cell_features)[0])
        
        # Get feature contributions
        feature_values = {}
        for col in feature_df.columns:
            feature_values[col] = {
                "value": float(cell[col]) if col in cell else 0.0,
                "description": get_feature_description(col)
            }
        
        return {
            "cell_id": int(cell_id),
            "risk_score": risk_score,
            "risk_level": get_risk_level(risk_score),
            "coordinates": {
                "geometry": str(cell.geometry) if 'geometry' in cell else None
            },
            "features": feature_values,
            "historical_data": {
                "is_blighted": bool(cell.get('is_blighted', False)),
                "target_blight_count": int(cell.get('target_blight_count', 0)),
                "overall_most_common_blight": str(cell.get('overall_most_common_blight', 'None')),
                "recent_most_common_blight": str(cell.get('recent_most_common_blight', 'None'))
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_cell_details: {e}")
        raise HTTPException(status_code=500, detail=f"Cell details calculation failed: {str(e)}")

def get_risk_level(risk_score: float) -> str:
    """Convert risk score to human-readable risk level."""
    if risk_score >= 0.8:
        return "Very High"
    elif risk_score >= 0.6:
        return "High"
    elif risk_score >= 0.4:
        return "Medium"
    elif risk_score >= 0.2:
        return "Low"
    else:
        return "Very Low"

@app.get("/api/top-risk-areas")
def get_top_risk_areas(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get the top highest-risk areas with their details.
    """
    try:
        # Calculate risk scores for all cells
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        columns_to_drop = [
            'cell_id', 'is_blighted', 'target_blight_count',
            'overall_most_common_blight', 'recent_most_common_blight'
        ]
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        X_features = df.drop(columns=existing_columns)
        
        risk_scores = model.predict(X_features)
        
        # Add risk scores to dataframe and sort
        df_with_risk = df.copy()
        df_with_risk['risk_score'] = risk_scores
        df_with_risk = df_with_risk.sort_values('risk_score', ascending=False).head(limit)
        
        top_areas = []
        for _, row in df_with_risk.iterrows():
            risk_score_val = float(row['risk_score'])
            top_areas.append({
                "cell_id": int(row['cell_id']),
                "risk_score": risk_score_val,
                "risk_level": get_risk_level(risk_score_val),
                "total_complaints_mean": float(row.get('total_complaints_mean') or 0),
                "blight_complaints_mean": float(row.get('blight_complaints_mean') or 0),
                "is_blighted": bool(row.get('is_blighted', False))
            })
        
        return top_areas
        
    except Exception as e:
        print(f"Error in get_top_risk_areas: {e}")
        raise HTTPException(status_code=500, detail=f"Top risk areas calculation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)