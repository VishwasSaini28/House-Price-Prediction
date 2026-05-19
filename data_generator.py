# data_generator.py
import pandas as pd
import numpy as np

# --- Step 1: Set number of samples ---
num_samples = 500
np.random.seed(42)  # reproducibility

# --- Step 2: Property types ---
property_types = ["Villa", "House", "Flat", "Apartment", "Building", "Empty Land", "Farm House"]
property_type = np.random.choice(property_types, num_samples)

# --- Step 3: Generate synthetic features ---
area_sqft = np.random.randint(500, 4000, num_samples)
bedrooms = np.random.randint(1, 6, num_samples)
bathrooms = np.random.randint(1, 4, num_samples)
parking_space = np.random.randint(0, 4, num_samples)
furnished = np.random.choice([0, 1], num_samples)
furniture_available = np.random.choice([0, 1], num_samples)
locality_score = np.random.randint(1, 11, num_samples)  # 1–10 rating
access_main_road = np.random.choice([0, 1], num_samples)
shopping_complex = np.random.choice([0, 1], num_samples)
local_market = np.random.choice([0, 1], num_samples)

# Conditional features
swimming_pool = []
gym_access = []
floor = []

for pt in property_type:
    # Swimming pool only for House, Villa, Farm House
    if pt in ["House", "Villa", "Farm House"]:
        swimming_pool.append(np.random.choice([0, 1]))
    else:
        swimming_pool.append(0)

    # Gym access only for Building, Apartment, Villa
    if pt in ["Building", "Apartment", "Villa"]:
        gym_access.append(np.random.choice([0, 1]))
    else:
        gym_access.append(0)

    # Floors only for House, Building, Flat, Apartment
    if pt == "House":
        floor.append(np.random.randint(1, 4))
    elif pt == "Building":
        floor.append(np.random.randint(1, 21))
    elif pt in ["Flat", "Apartment"]:
        floor.append(np.random.randint(1, 31))
    else:
        floor.append(0)

# --- Step 4: Generate synthetic target (price) ---
price = (
    50000
    + area_sqft * 50
    + bedrooms * 20000
    + bathrooms * 15000
    + np.array(swimming_pool) * 50000
    + parking_space * 15000
    + furnished * 50000
    + furniture_available * 30000
    + locality_score * 10000
    + access_main_road * 25000
    + shopping_complex * 20000
    + local_market * 15000
    + np.array(gym_access) * 20000
    + np.array(floor) * 5000
    + np.random.randint(-30000, 30000, num_samples)  # noise
)

# --- Step 5: Create DataFrame ---
df = pd.DataFrame({
    "property_type": property_type,
    "area_sqft": area_sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "swimming_pool": swimming_pool,
    "gym_access": gym_access,
    "floor": floor,
    "parking_space": parking_space,
    "furnished": furnished,
    "furniture_available": furniture_available,
    "locality_score": locality_score,
    "access_main_road": access_main_road,
    "shopping_complex": shopping_complex,
    "local_market": local_market,
    "price": price
})

# --- Step 6: Save to CSV ---
df.to_csv("house_data.csv", index=False)
print("Synthetic dataset saved as house_data.csv with", num_samples, "rows.")
