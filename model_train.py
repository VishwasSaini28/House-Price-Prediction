# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# --- Step 1: Load dataset ---
df = pd.read_csv("house_data.csv")

# --- Step 2: Define features (X) and target (y) ---
X = df[[
    "property_type", "area_sqft", "bedrooms", "bathrooms", "swimming_pool",
    "gym_access", "floor", "parking_space", "furnished", "furniture_available",
    "locality_score", "access_main_road", "shopping_complex", "local_market"
]]
y = df["price"]

# --- Step 3: Preprocessing ---
numeric_features = [
    "area_sqft", "bedrooms", "bathrooms", "floor",
    "parking_space", "locality_score"
]
categorical_features = ["property_type"]
binary_features = [
    "swimming_pool", "gym_access", "furnished", "furniture_available",
    "access_main_road", "shopping_complex", "local_market"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ("bin", "passthrough", binary_features)
    ]
)

# --- Step 4: Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Step 5: Build pipeline ---
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# --- Step 6: Train model ---
model.fit(X_train, y_train)

# --- Step 7: Evaluate model ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Training complete!")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# --- Step 8: Save trained model ---
joblib.dump(model, "house_price_model.pkl")
print("Model saved as house_price_model.pkl")

# --- Step 9: Test prediction with new property ---
new_property = pd.DataFrame(
    [["Villa", 2200, 3, 2, 1, 1, 2, 2, 1, 1, 7, 1, 1, 1]],
    columns=[
        "property_type", "area_sqft", "bedrooms", "bathrooms", "swimming_pool",
        "gym_access", "floor", "parking_space", "furnished", "furniture_available",
        "locality_score", "access_main_road", "shopping_complex", "local_market"
    ]
)

predicted_price = model.predict(new_property)
print("Predicted Price for new property:", predicted_price[0])
