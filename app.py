# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load the Saved Model Components ---
try:
    model = joblib.load('final_sales_model.pkl')
    scaler = joblib.load('final_scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
except FileNotFoundError:
    st.error("Model files not found! Please run 'save_model.py' first.")
    st.stop()

# --- 2. App Title and Description ---
st.set_page_config(page_title="Internboot Sales AI", layout="centered")
st.image("https://cdn-icons-png.flaticon.com/512/3094/3094918.png", width=100) # Optional Icon
st.title("Store Sales Forecaster 📈")
st.markdown("""
This app uses **Lasso Regression** to predict daily store sales based on date, promotion, and economic factors.
""")

st.write("---")

# --- 3. User Input Section ---
st.sidebar.header("Input Parameters")

# A. Store & Product Details
store_nbr = st.sidebar.number_input("Store Number (1-54)", min_value=1, max_value=54, value=1)
# Create a list of families for the dropdown (simplified)
product_families = [
    'AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS', 'BREAD/BAKERY', 
    'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS', 'FROZEN FOODS', 
    'GROCERY I', 'GROCERY II', 'HARDWARE', 'HOME AND KITCHEN I', 
    'HOME AND KITCHEN II', 'HOME APPLIANCES', 'HOME CARE', 'LADIESWEAR', 
    'LAWN AND GARDEN', 'LINGERIE', 'LIQUOR,WINE,BEER', 'MAGAZINES', 'MEATS', 
    'PERSONAL CARE', 'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'POULTRY', 
    'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD'
]
family = st.sidebar.selectbox("Product Category", product_families, index=12) # Default to GROCERY I

# B. Date & Promotions
date_input = st.sidebar.date_input("Select Date")
onpromotion = st.sidebar.number_input("Items on Promotion", min_value=0, value=0)
oil_price = st.sidebar.slider("Oil Price ($)", min_value=30.0, max_value=120.0, value=80.0)

# --- 4. Prediction Logic ---
if st.button("Predict Sales Output"):
    
    # A. Preprocess the Input Data to match Training Data
    # Extract date features
    input_data = {
        'store_nbr': store_nbr,
        'onpromotion': onpromotion,
        'dcoilwtico': oil_price,
        'year': date_input.year,
        'month': date_input.month,
        'day': date_input.day,
        'day_of_week': date_input.weekday()
    }
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # One-Hot Encode the 'family' selection manually
    # We create a dataframe with all 0s for the family columns
    input_df = pd.get_dummies(input_df) # This won't generate all columns since we have 1 row
    
    # CRITICAL STEP: Align with model columns
    # We create a new DF with the exact columns the model expects, filling missing ones with 0
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # Set the specific family column to 1
    family_col = f"family_{family}"
    if family_col in input_df.columns:
        input_df[family_col] = 1

    # B. Scale the Input
    input_scaled = scaler.transform(input_df)
    
    # C. Predict
    prediction = model.predict(input_scaled)[0]
    
    # Ensure no negative sales predictions
    final_prediction = max(0, prediction)

    # --- 5. Display Results ---
    st.success(f"## 💰 Predicted Sales: ${final_prediction:,.2f}")
    
    # --- 6. Visualization (Requirements Requirement) ---
    st.write("### Forecast Visualization")
    # Generating dummy historical context for the chart 
    chart_data = pd.DataFrame({
        'Scenario': ['Optimistic', 'Predicted', 'Pessimistic'],
        'Sales': [final_prediction * 1.2, final_prediction, final_prediction * 0.8]
    })
    
    st.bar_chart(chart_data.set_index('Scenario'))
    
    st.info(f"Prediction based on Store {store_nbr}, {family}, Oil Price: ${oil_price}")
