import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = "../../dataset/student_placement.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

X = df.drop("Placement", axis=1)
y = df["Placement"]


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 5. LOAD MODEL
# ==========================================

model = joblib.load("placement_model.pkl")


# ==========================================
# 6. TRAIN AND TEST ACCURACY
# ==========================================

train_accuracy = model.score(
    X_train_scaled,
    y_train
)

test_accuracy = model.score(
    X_test_scaled,
    y_test
)


# ==========================================
# 7. PRINT RESULTS
# ==========================================

print("\n========== OVERFITTING ANALYSIS ==========")

print(f"Training Accuracy : {train_accuracy:.2f}")
print(f"Testing Accuracy  : {test_accuracy:.2f}")

gap = train_accuracy - test_accuracy

print(f"Accuracy Gap      : {gap:.2f}")


# ==========================================
# 8. DIAGNOSIS
# ==========================================

print("\n========== DOCTOR'S DIAGNOSIS ==========")

if gap > 0.20:
    print("🚨 Possible Overfitting detected!")
    print("The model performs much better on training data than testing data.")
    print("Suggestions:")
    print("- Collect more training data")
    print("- Reduce model complexity")
    print("- Use cross-validation")
    print("- Apply regularization")
    
elif gap > 0.10:
    print("⚠️ Mild overfitting may be present.")
    print("Consider collecting more data or tuning the model.")
    
else:
    print("✅ No significant overfitting detected.")
    print("Training and testing performance are relatively close.")


print("\n========== ANALYSIS COMPLETE ==========")