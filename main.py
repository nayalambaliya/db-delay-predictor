from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class TrainQuery(BaseModel):
    hour: int
    day_of_week: int  # 0=Monday, 6=Sunday
    month: int
    category: int    # 1 to 5

@app.post("/predict")
def predict(query: TrainQuery):
    category_enc = query.category - 1  # map 1-5 to 0-4
    features = [[query.hour, query.day_of_week, query.month, category_enc]]
    prob = model.predict_proba(features)[0][1]
    return {
        "delay_probability": round(prob * 100, 1),
        "verdict": "Likely delayed 🚨" if prob > 0.5 else "Probably on time ✅"
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")
