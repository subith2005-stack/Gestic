import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
import pickle
import os

# Dataset location
DATA_PATH = "data/raw/gesture_data.csv"

# Model save location
MODEL_PATH = "models/gesture_model.pkl"

# Ensure model folder exists
os.makedirs("models", exist_ok=True)

print("Loading dataset...")

# Load dataset
data = pd.read_csv(DATA_PATH)

# Separate features and labels
X = data.drop("gesture", axis=1)
y = data["gesture"]

print("Dataset shape:", data.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

# Initialize classifier
model = RandomForestClassifier(n_estimators=100)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("Model saved to:", MODEL_PATH)
