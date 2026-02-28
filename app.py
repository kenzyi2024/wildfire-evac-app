import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the Models
@st.cache_resource # This caches the models so they don't reload on every button click
def load_models():
    rsf = joblib.load('wildfire_rsf_model.pkl')
    gbsa = joblib.load('wildfire_gbsa_model.pkl')
    return rsf, gbsa

rsf, gbsa = load_models()

# 2. Recreate the Feature Engineering Function
def engineer_features(df):
    df = df.copy()
    df['dist_to_5km_threshold_m'] = np.maximum(df['dist_min_ci_0_5h'] - 5000, 0)
    safe_speed = np.maximum(df['closing_speed_m_per_h'], 0.1)
    df['eta_hours'] = df['dist_to_5km_threshold_m'] / safe_speed
    df['danger_index'] = (df['area_first_ha'] * df['area_growth_rate_ha_per_h']) / (df['dist_min_ci_0_5h'] + 1)
    df['directed_threat_speed'] = df['centroid_speed_m_per_h'] * df['alignment_abs']
    return df

# 3. Build the User Interface
st.title("🔥 Wildfire Evacuation Threat Predictor")
st.write("Enter the early tracking signals (first 5 hours) to predict the probability of a fire hitting an evacuation zone.")

st.sidebar.header("Fire Telemetry Inputs")

# Create input sliders for the most important features
dist_min = st.sidebar.slider("Distance to Evac Zone (meters)", 5000, 100000, 15000)
area_first = st.sidebar.slider("Initial Area (hectares)", 0.0, 5000.0, 50.0)
area_growth_rate = st.sidebar.slider("Area Growth Rate (ha/hr)", 0.0, 1000.0, 10.0)
closing_speed = st.sidebar.slider("Closing Speed (m/hr)", -500.0, 5000.0, 200.0)
alignment = st.sidebar.slider("Alignment to Zone (0 to 1)", 0.0, 1.0, 0.8)
centroid_speed = st.sidebar.slider("Centroid Speed (m/hr)", 0.0, 5000.0, 250.0)

# 4. Process the Input when the user clicks "Predict"
if st.button("Predict Threat Level"):
    # Create a dataframe with the user's inputs (filling others with safe baseline medians for the demo)
    input_data = pd.DataFrame({
        'dist_min_ci_0_5h': [dist_min],
        'log1p_area_first': [np.log1p(area_first)],
        'num_perimeters_0_5h': [3], # Dummy median
        'dt_first_last_0_5h': [4.5], # Dummy median
        'event_start_month': [8], # August
        'alignment_abs': [alignment],
        'low_temporal_resolution_0_5h': [0],
        'area_first_ha': [area_first],
        'event_start_dayofweek': [3],
        'event_start_hour': [14],
        'cross_track_component': [0],
        'radial_growth_m': [100],
        'area_growth_rate_ha_per_h': [area_growth_rate],
        'closing_speed_m_per_h': [closing_speed],
        'centroid_speed_m_per_h': [centroid_speed]
    })
    
    # Apply engineering
    processed_data = engineer_features(input_data)
    
    # Make sure we only feed the exact columns the model expects
    expected_cols = [
        'dist_min_ci_0_5h', 'log1p_area_first', 'num_perimeters_0_5h', 
        'dt_first_last_0_5h', 'event_start_month', 'alignment_abs', 
        'low_temporal_resolution_0_5h', 'area_first_ha', 'event_start_dayofweek', 
        'event_start_hour', 'cross_track_component', 'radial_growth_m',
        'dist_to_5km_threshold_m', 'eta_hours', 'danger_index', 'directed_threat_speed'
    ]
    final_X = processed_data[expected_cols]

    # Generate Predictions
    horizons = [12, 24, 48, 72]
    
    # RSF
    fn_rsf = rsf.predict_survival_function(final_X)[0]
    prob_rsf = 1 - fn_rsf(np.clip(horizons, fn_rsf.domain[0], fn_rsf.domain[1]))
    
    # GBSA
    fn_gbsa = gbsa.predict_survival_function(final_X)[0]
    prob_gbsa = 1 - fn_gbsa(np.clip(horizons, fn_gbsa.domain[0], fn_gbsa.domain[1]))
    
    # Ensemble
    prob_ensemble = (prob_rsf + prob_gbsa) / 2.0
    
    # 5. Display the Results
    st.subheader("⚠️ Threat Forecast")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("12 Hours", f"{prob_ensemble[0]*100:.1f}%")
    col2.metric("24 Hours", f"{prob_ensemble[1]*100:.1f}%")
    col3.metric("48 Hours", f"{prob_ensemble[2]*100:.1f}%")
    col4.metric("72 Hours", f"{prob_ensemble[3]*100:.1f}%")
    
    # Plot a nice chart
    chart_data = pd.DataFrame({
        "Hours": horizons,
        "Probability of Impact": prob_ensemble * 100
    }).set_index("Hours")
    st.line_chart(chart_data)