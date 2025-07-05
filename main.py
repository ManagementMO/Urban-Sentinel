from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dataset_for_statelite.main import check_if_darker_over_years

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)