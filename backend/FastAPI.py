from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend (React) to access the backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def get_users():
    return [{"id": 1, "name": "Anshuman"}]
