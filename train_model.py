import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib
import os

# ------------------ SETUP ------------------
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

DATA_PATH = "data/Telco-Customer-Churn.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("❌ Dataset not found! Please put 'Telco-Customer-Churn.csv' in the data/ folder.")

# ------------------ LOAD DATA ------------------
df = pd.read_csv(DATA_PATH)

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# ------------------ ENCODING ------------------
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    if col != "Churn":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ------------------ SPLIT + TRAIN ------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ------------------ SAVE FILES ------------------
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")
joblib.dump(list(X.columns), "models/feature_columns.pkl")

print("✅ Model trained successfully!")
print(f"📂 Saved files in 'models/' ({len(X.columns)} features)")
