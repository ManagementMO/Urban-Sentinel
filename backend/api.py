import joblib
import geopandas as gpd
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ==============================================================================
# --- APPLICATION SETUP ---
# ==============================================================================

# Create the FastAPI application instance
app = FastAPI(title="Urban Sentinel API")

# --- CORS Middleware ---
# This allows your React frontend (running on a different address)
# to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Load Your Assets (Model and Data) ---
# This happens only ONCE when the API starts up.
print("Loading model and data assets...")
model = joblib.load("urban_sentinel_model.pkl")
gdf = gpd.read_file("toronto_grid_with_temporal_features.geojson")
print("Assets loaded successfully!")


# ==============================================================================
# --- API ENDPOINTS ---
# ==============================================================================

@app.get("/")
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Welcome to the Urban Sentinel Predictive Blight API"}


@app.get("/api/predict-risk")
def predict_risk():
    """
    This is the main endpoint for your application.
    It uses the trained model to predict a risk score for every grid cell
    and returns the full GeoJSON with this new information.
    """
    print("Received request to predict risk scores...")

    # Step 1: Prepare the feature data (X) exactly as you did for training
    df = pd.DataFrame(gdf.drop(columns='geometry'))
    columns_to_drop = [
        'cell_id', 'is_blighted', 'target_blight_count',
        'overall_most_common_blight', 'recent_most_common_blight'
    ]
    X_features = df.drop(columns=columns_to_drop)

    # Step 2: Use the model to predict probabilities (the "risk score")
    # model.predict_proba gives two columns: [prob_of_0, prob_of_1]
    # We want the probability of being blighted (class 1).
    risk_scores = model.predict_proba(X_features)[:, 1]
    
    # Step 3: Add this new risk score to our original GeoDataFrame
    gdf['risk_score'] = risk_scores
    print(f"  - Calculated risk scores for {len(gdf)} cells.")

    # Step 4: Return the entire GeoJSON.
    # FastAPI will automatically convert this to a JSON response.
    # Note: In a production app with huge data, you'd convert this to a more
    # efficient format, but for a hackathon, this is perfect.
    return gdf.to_dict(orient='records')