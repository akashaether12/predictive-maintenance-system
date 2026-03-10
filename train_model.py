import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------
# 1. Load Dataset
# ------------------------------

data = pd.read_csv("predictive_maintenance.csv")

print("Dataset Loaded Successfully")
print(data.head())

# ------------------------------
# 2. Remove Unnecessary Columns
# ------------------------------

data = data.drop(["UDI", "Product ID", "Failure Type"], axis=1)

# ------------------------------
# 3. Encode Machine Type
# ------------------------------

encoder = LabelEncoder()
data["Type"] = encoder.fit_transform(data["Type"])

# ------------------------------
# 4. Split Features and Target
# ------------------------------

X = data.drop("Target", axis=1)
y = data["Target"]

# ------------------------------
# 5. Train Test Split
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ------------------------------
# 6. Feature Scaling
# ------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# 7. Train Machine Learning Model
# ------------------------------

model = RandomForestClassifier()

model.fit(X_train, y_train)

# ------------------------------
# 8. Evaluate Model
# ------------------------------

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ------------------------------
# 9. User Input for New Machine
# ------------------------------

print("\nEnter Machine Sensor Values")

type_val = int(input("Machine Type (0=L, 1=M, 2=H): "))
air_temp = float(input("Air Temperature [K]: "))
process_temp = float(input("Process Temperature [K]: "))
rpm = float(input("Rotational Speed [rpm]: "))
torque = float(input("Torque [Nm]: "))
tool_wear = float(input("Tool Wear [min]: "))

# Create input array
machine = pd.DataFrame([[
    type_val,
    air_temp,
    process_temp,
    rpm,
    torque,
    tool_wear
]], columns=X.columns)

machine_scaled = scaler.transform(machine)

# Predict
prediction = model.predict(machine_scaled)

# Probability
probability = model.predict_proba(machine_scaled)

# ------------------------------
# 10. Result
# ------------------------------

print("\nPrediction Result")

if prediction[0] == 1:
    print("⚠️ Machine likely to FAIL - Maintenance Required")
else:
    print("✅ Machine is Healthy")

print("Failure Probability:", probability[0][1])