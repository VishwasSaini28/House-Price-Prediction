import streamlit as st
import joblib
import pandas as pd

# --- Load models ---
reg_model = joblib.load("house_price_model.pkl")
clf_model = joblib.load("house_trend_model.pkl")

# --- Page config ---
st.set_page_config(page_title="🏠 Property Price Estimator", layout="wide")
st.title("🏠 Property Price Estimation")

# --- Step control ---
if "step" not in st.session_state:
    st.session_state.step = 1

# --- Step 1: Select property type ---
if st.session_state.step == 1:
    property_type = st.selectbox("Property Type", 
        ["Villa", "House", "Flat", "Apartment", "Building", "Empty Land", "Farm House"])
    if st.button("Next ➡️"):
        st.session_state.property_type = property_type
        st.session_state.step = 2

# --- Step 2: Enter property details ---
elif st.session_state.step == 2:
    property_type = st.session_state.property_type
    st.subheader(f"Selected Property Type: **{property_type}**")

    with st.form("details_form"):
        area_sqft = st.number_input("Area (sqft)", 500, 750000, 5000, step=100)
        bedrooms = st.slider("Bedrooms", 0, 10, 3)
        bathrooms = st.slider("Bathrooms", 0, 6, 2)
        parking_space = st.slider("Parking Spaces", 0, 6, 2)
        furnished = st.checkbox("Furnished")
        furniture_available = st.checkbox("Furniture Available")
        locality_score = st.slider("Locality Score (1-10)", 1, 10, 7)
        access_main_road = st.checkbox("Access to Main Road")
        shopping_complex = st.checkbox("Nearby Shopping Complex")
        local_market = st.checkbox("Nearby Local Market")
        previous_price = st.number_input("Previous Price (₹)", 100000, 5000000, 500000, step=50000)

        floor = 0
        if property_type == "House":
            floor = st.slider("Floors (House)", 1, 3, 2)
        elif property_type == "Building":
            floor = st.slider("Floors (Building)", 1, 20, 5)
        elif property_type in ["Flat", "Apartment"]:
            floor = st.slider("Floor (Flat/Apartment)", 1, 30, 2)

        swimming_pool = False
        if property_type in ["House", "Villa", "Farm House"]:
            swimming_pool = st.checkbox("Swimming Pool")

        gym_access = False
        if property_type in ["Building", "Apartment", "Villa"]:
            gym_access = st.checkbox("Gym Access")

        residence_status = st.selectbox("Residence Status", ["Residence", "Non-Residence"])
        registration_status = st.selectbox("Registration Status", ["Registrable", "Documented"])

        submitted = st.form_submit_button("Predict Price")

    if submitted:
        st.session_state.details = {
            "area_sqft": area_sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "swimming_pool": int(swimming_pool),
            "parking_space": parking_space,
            "furnished": int(furnished),
            "furniture_available": int(furniture_available),
            "locality_score": locality_score,
            "access_main_road": int(access_main_road),
            "shopping_complex": int(shopping_complex),
            "local_market": int(local_market),
            "previous_price": previous_price,
            "floor": floor,
            "property_type": property_type,
            "residence_status": residence_status,
            "registration_status": registration_status,
            "gym_access": int(gym_access)
        }
        st.session_state.step = 3

# --- Step 3: Show prediction results ---
elif st.session_state.step == 3:
    st.subheader("📊 Prediction Results")

    new_property = pd.DataFrame([st.session_state.details])

    predicted_price = reg_model.predict(new_property)[0]
    predicted_trend = clf_model.predict(new_property)[0]

    st.success(f"💰 Estimated Price: ₹{predicted_price:,.0f}")
    if predicted_trend == "Up":
        st.markdown("<span style='color:green; font-size:20px;'>⬆️ Price Trend: UP</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:red; font-size:20px;'>⬇️ Price Trend: DOWN</span>", unsafe_allow_html=True)

    st.markdown("### 🏡 Property Summary")
    st.write(new_property)

    if st.button("🔄 Predict Another Property"):
        st.session_state.step = 1
        st.session_state.details = {}
