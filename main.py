from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dataset_for_statelite.main import check_if_darker_over_years
from typing import List
import random

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Urban Sentinel API - Use /{x}/{y} to analyze pixel brightness over time"}

@app.get("/{x}/{y}")
def analyze_pixel(x: int, y: int):
    try:
        result = check_if_darker_over_years((x, y))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing pixel: {str(e)}")

@app.get("/decay")
def get_decay_data():
    """
    Returns a list of decay data points for the frontend map.
    In a real implementation, this would query a database or process actual data.
    """
    # Generate sample decay data around Toronto
    # Toronto bounds approximately: 43.58 to 43.86 lat, -79.64 to -79.11 lon
    decay_points = []
    
    # Known areas with different decay levels (sample data)
    sample_locations = [
        # Downtown core - mixed decay levels
        {"lat": 43.6532, "lon": -79.3832, "decay": 0.3, "source": "311"},
        {"lat": 43.6426, "lon": -79.3871, "decay": 0.6, "source": "satellite"},
        {"lat": 43.6488, "lon": -79.3965, "decay": 0.8, "source": "combined"},
        
        # East end
        {"lat": 43.6763, "lon": -79.2930, "decay": 0.7, "source": "311"},
        {"lat": 43.6689, "lon": -79.3147, "decay": 0.5, "source": "satellite"},
        
        # West end
        {"lat": 43.6369, "lon": -79.4780, "decay": 0.9, "source": "combined"},
        {"lat": 43.6534, "lon": -79.4635, "decay": 0.4, "source": "311"},
        
        # North York
        {"lat": 43.7615, "lon": -79.4111, "decay": 0.3, "source": "satellite"},
        {"lat": 43.7701, "lon": -79.4125, "decay": 0.6, "source": "combined"},
        
        # Scarborough
        {"lat": 43.7731, "lon": -79.2570, "decay": 0.7, "source": "311"},
        {"lat": 43.7635, "lon": -79.1887, "decay": 0.5, "source": "satellite"},
        
        # Etobicoke
        {"lat": 43.6205, "lon": -79.5132, "decay": 0.8, "source": "combined"},
        {"lat": 43.6435, "lon": -79.5655, "decay": 0.4, "source": "311"},
    ]
    
    for loc in sample_locations:
        decay_points.append({
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "decay_level": loc["decay"],
            "source": loc["source"]
        })
    
    # Add some random points for density
    for _ in range(20):
        decay_points.append({
            "latitude": round(random.uniform(43.58, 43.86), 4),
            "longitude": round(random.uniform(-79.64, -79.11), 4),
            "decay_level": round(random.uniform(0.1, 0.9), 2),
            "source": random.choice(["311", "satellite", "combined"])
        })
    
    return decay_points

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)