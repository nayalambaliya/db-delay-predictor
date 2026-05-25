import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load data
print("Loading data...")
df = pd.read_csv("data/DBtrainrides.csv", low_memory=False)

# Parse datetime and extract features
df["arrival_plan"] = pd.to_datetime(df["arrival_plan"])
df["hour"] = df["arrival_plan"].dt.hour
df["day_of_week"] = df["arrival_plan"].dt.dayofweek
df["month"] = df["arrival_plan"].dt.month

# Target: 1 if delayed more than 5 mins
df["delayed"] = (df["arrival_delay_m"] > 5).astype(int)

# Drop rows with missing values in our columns
df = df.dropna(subset=["hour", "day_of_week", "month", "category", "delayed"])

# Encode category (train type) as number
df["category_enc"] = df["category"].astype("category").cat.codes

features = ["hour", "day_of_week", "month", "category_enc"]
X = df[features]
y = df["delayed"]

print(f"Dataset size: {len(df)} rows")
print(f"Delay rate: {y.mean()*100:.1f}%")

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
print("Training model...")
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test)*100:.1f}%")

# Save model + category mapping
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

category_map = dict(enumerate(df["category"].astype("category").cat.categories))
with open("category_map.pkl", "wb") as f:
    pickle.dump(category_map, f)

print("Saved model.pkl ✅")
print("Categories:", category_map)
