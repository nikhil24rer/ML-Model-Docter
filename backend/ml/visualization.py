import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = "../../dataset/student_placement.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")


# ==========================================
# 2. CREATE VISUALIZATION FOLDER
# ==========================================

import os

os.makedirs("../../visualizations", exist_ok=True)


# ==========================================
# 3. FEATURE DISTRIBUTION
# ==========================================

numeric_columns = df.select_dtypes(
    include="number"
).columns

for column in numeric_columns:

    plt.figure()

    plt.hist(df[column], bins=10)

    plt.title(f"{column} Distribution")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.savefig(
        f"../../visualizations/{column}_distribution.png"
    )

    plt.close()


# ==========================================
# 4. LOAD MODEL
# ==========================================

X = df.drop("Placement", axis=1)
y = df["Placement"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = joblib.load("scaler.pkl")
model = joblib.load("placement_model.pkl")

X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)


# ==========================================
# 5. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

display.plot()

plt.title("Confusion Matrix")

plt.savefig(
    "../../visualizations/confusion_matrix.png"
)

plt.close()


# ==========================================
# 6. FEATURE IMPORTANCE
# ==========================================

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure()

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "../../visualizations/feature_importance.png"
)

plt.close()


print("\n========== VISUALIZATION COMPLETE ==========")

print("Graphs saved inside:")
print("visualizations/")