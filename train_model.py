# train_model.py

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
import os

# Load data
data = load_iris()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Save model to model/model.joblib
dump(model, "model/model.joblib")

print("✅ Model trained and saved to 'model/model.joblib'")
