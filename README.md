# 🚆 DB Delay Predictor

A machine learning web app that predicts the probability of a Deutsche Bahn train being delayed, based on 1.85 million real ride records from NRW, Germany.

## Demo

Enter a departure hour, day of week, month, and train category — the model returns the delay probability in real time.

## Tech Stack

- **ML Model** — Random Forest Classifier (scikit-learn)
- **Backend** — FastAPI (Python)
- **Frontend** — Vanilla HTML/CSS/JS
- **Data** — 1.85M Deutsche Bahn rides from NRW (2024)

## How it works

1. Raw CSV data is loaded and parsed using pandas
2. Features extracted: hour, day of week, month, train category
3. Target: binary classification — delayed if arrival delay > 5 minutes
4. Random Forest trained on 80% of data, evaluated on 20%
5. Model achieves **94% accuracy** (note: 6% base delay rate — class imbalance acknowledged)
6. Trained model serialized with pickle and served via FastAPI REST API
7. Frontend communicates with `/predict` endpoint and displays result

## Run locally

```bash
git clone https://github.com/nayalambaliya/db-delay-predictor.git
cd db-delay-predictor

python3 -m venv venv
source venv/bin/activate
pip install pandas scikit-learn fastapi uvicorn

# Add your dataset to data/DBtrainrides.csv
python3 train_model.py
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`

## What I learned

- Handling real-world imbalanced datasets
- Building and serializing ML models for production
- Connecting an ML backend to a REST API with FastAPI
- End-to-end deployment of a data-driven web application
