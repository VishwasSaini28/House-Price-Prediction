# test_model.py
import pandas as pd
import joblib

# --- Step 1: Load trained model ---
model = joblib.load("house_price_model.pkl")

# --- Step 2: Define test properties ---
# Each row: [property_type, area_sqft, bedrooms, bathrooms, swimming_pool,
#            gym_access, floor, parking_space, furnished, furniture_available,
#            locality_score, access_main_road, shopping_complex, local_market]

test_properties = pd.DataFrame(
    [
        # Villa with pool and gym
        ["Villa", 2200, 3, 2, 1, 1, 2, 2, 1, 1, 7, 1, 1, 1],
        # Apartment (no pool, gym access)
        ["Apartment", 1500, 2, 1, 0, 1, 5, 1, 0, 0, 6, 1, 0, 1],
        # Building (gym, multiple floors, no pool)
        ["Building", 3500, 5, 3, 0, 1, 10, 3, 1, 1, 9, 1, 1, 1],
        # Empty Land (minimal features)
        ["Empty Land", 2000, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0],
    ],
    columns=[
        "property_type", "area_sqft", "bedrooms", "bathrooms", "swimming_pool",
        "gym_access", "floor", "parking_space", "furnished", "furniture_available",
        "locality_score", "access_main_road", "shopping_complex", "local_market"
    ]
)

# --- Step 3: Predict prices ---
predictions = model.predict(test_properties)

# --- Step 4: Display results ---
for i, price in enumerate(predictions):
    print(f"Property {i+1} ({test_properties.iloc[i]['property_type']}) predicted price: ₹{price:,.0f}")
