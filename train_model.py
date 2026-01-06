import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "student__data.csv")

df = pd.read_csv(DATA_PATH)




X = df[["Study_Hours", "Attendance", "Previous_Score"]]
y = df["Final_Score"]

model = LinearRegression()
model.fit(X, y)

os.makedirs("model", exist_ok=True)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")
