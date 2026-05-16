"""
NHS OPEL Level Predictor Dashboard
Local Streamlit application for operational pressure forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="NHS OPEL Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== NHS BRANDING & STYLING ==============
st.markdown("""
    <style>
    .header {
        background: linear-gradient(135deg, #005EB8 0%, #003087 100%);
        color: white;
        padding: 25px;
        border-radius: 8px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header h1 {
        margin: 0;
        font-size: 2.5em;
    }
    .header p {
        margin: 5px 0 0 0;
        font-size: 1.1em;
        opacity: 0.9;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 8px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .opel1 { 
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        color: #2e7d32;
        border-left: 6px solid #009639;
    }
    .opel2 { 
        background: linear-gradient(135deg, #fff9c4 0%, #ffeb3b 100%);
        color: #f57f17;
        border-left: 6px solid #FFB81C;
    }
    .opel3 { 
        background: linear-gradient(135deg, #ffe0b2 0%, #ffcc80 100%);
        color: #e65100;
        border-left: 6px solid #ED8B00;
    }
    .opel4 { 
        background: linear-gradient(135deg, #ffebee 0%, #ef9a9a 100%);
        color: #c62828;
        border-left: 6px solid #DA291C;
    }
    .metric-box {
        background: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #005EB8;
    }
    .info-box {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #1976d2;
        margin: 15px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============== HEADER ==============
st.markdown('''
    <div class="header">
        <h1>🏥 NHS OPEL Level Predictor</h1>
        <p>AI-powered operational pressure forecasting for decision support</p>
    </div>
''', unsafe_allow_html=True)

# ============== SIDEBAR INFO ==============
with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.markdown("""
    **Model:** Logistic Regression Classifier
    
    **Accuracy:** 92.0%
    
    **OPEL 4 Recall:** 90.0%
    
    **Features:** 19 operational, workforce & environmental
    
    **Training data:** 2,190 daily observations (3 trusts, 2 years)
    
    **Framework:** scikit-learn
    """)
    
    st.markdown("### ⚙️ How to use")
    st.markdown("""
    1. Enter today's operational metrics
    2. Model predicts OPEL level
    3. Review confidence scores
    4. Check feature importance
    5. Use for decision support only
    """)
    
    st.markdown("### ⚠️ Important")
    st.markdown("""
    This is a **decision support tool**, not autonomous decision-making. Always combine with human clinical judgement.
    """)

# ============== LOAD MODEL ==============
@st.cache_resource
def load_model():
    try:
        model = joblib.load('nhs_opel_baseline_model.pkl')
        scaler = joblib.load('nhs_opel_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure nhs_opel_baseline_model.pkl and nhs_opel_scaler.pkl are in the same directory as this script.")
        st.stop()

model, scaler = load_model()
st.success("✓ Model and scaler loaded successfully")

# ============== FEATURE NAMES ==============
feature_names = [
    'flu_rate_per_100k', 'temperature_c', 'gp_appointment_lag_days',
    'ae_attendance', 'ae_4hr_breach_pct', 'ae_mean_wait_mins',
    'ae_12hr_trolley_waits', 'social_care_delay_days', 'nctr_patients',
    'bed_occupancy_pct', 'nurse_sickness_pct', 'unfilled_shifts',
    'ambulance_handover_avg_mins', 'ambulance_handover_over_60min',
    'red_flag_incidents', 'ae_pressure_index', 'bed_social_care_ratio',
    'workforce_stress_index', 'system_flow_score'
]

# ============== INPUT FORM ==============
st.header("📋 Enter Today's Operational Data")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Environmental & Demand**")
    flu_rate = st.slider('Flu rate (per 100k)', 0.0, 100.0, 15.0, step=0.5, help="Current flu incidence")
    temp = st.slider('Temperature (°C)', -5.0, 25.0, 10.0, step=0.5, help="Local ambient temperature")
    gp_lag = st.slider('GP appointment lag (days)', 0.0, 20.0, 10.0, step=0.5, help="Average days to next appointment")
    
    st.markdown("**A&E Performance**")
    ae_att = st.slider('A&E attendance (daily)', 100, 500, 300, step=10, help="Total attendances today")
    ae_breach = st.slider('A&E 4hr breach rate (%)', 0.0, 1.0, 0.08, step=0.01, help="Proportion waiting >4 hours")
    ae_wait = st.slider('A&E mean wait (mins)', 60, 200, 130, step=5, help="Average wait across all patients")

with col2:
    st.markdown("**Bed & Capacity**")
    trolley = st.slider('A&E 12hr trolley waits (count)', 0, 20, 5, step=1, help="Patients waiting >12 hours on trolley")
    soc_care = st.slider('Social care delay (days)', 0.0, 30.0, 15.0, step=0.5, help="Average delay for discharge")
    nctr = st.slider('NCTR patients (medically fit)', 0, 100, 50, step=5, help="No criteria to reside in bed")
    bed_occ = st.slider('Bed occupancy (%)', 0.0, 1.2, 0.92, step=0.01, help="Current occupancy as proportion of available")
    
    st.markdown("**Workforce**")
    nurse_sick = st.slider('Nurse sickness (%)', 0.0, 0.2, 0.08, step=0.01, help="Proportion of nursing staff absent")
    unfilled = st.slider('Unfilled shifts (today)', 0, 20, 3, step=1, help="Number of shifts unable to staff")

with col3:
    st.markdown("**Patient Safety & Flow**")
    amb_handover = st.slider('Ambulance handover (mins)', 10, 30, 15, step=1, help="Average time from arrival to handover")
    amb_over60 = st.slider('Ambulance >60min handover (count)', 0, 10, 2, step=1, help="Handovers delayed >60 minutes")
    red_flags = st.slider('Red flag incidents (today)', 0, 10, 2, step=1, help="Patient safety incidents reported")

# ============== CALCULATE ENGINEERED FEATURES ==============
ae_pressure = (ae_breach * 0.4) + (ae_wait / 240 * 0.3) + (nctr / 50 * 0.3)
bed_soc_ratio = soc_care / bed_occ if bed_occ > 0 else 0
workforce_stress = (unfilled * 0.5) + (nurse_sick / 30 * 0.5)
system_flow = (bed_occ * 0.4) + (soc_care / 20 * 0.3) + (gp_lag / 14 * 0.3)

# ============== CREATE PREDICTION ==============
input_data = np.array([[
    flu_rate, temp, gp_lag, ae_att, ae_breach, ae_wait, trolley, soc_care, nctr,
    bed_occ, nurse_sick, unfilled, amb_handover, amb_over60, red_flags,
    ae_pressure, bed_soc_ratio, workforce_stress, system_flow
]])

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]
probabilities = model.predict_proba(input_scaled)[0]

# ============== DISPLAY PREDICTION ==============
st.header("🎯 Prediction Result")

opel_names = {
    1: 'OPEL 1 — Normal Operations',
    2: 'OPEL 2 — Escalated',
    3: 'OPEL 3 — Critical',
    4: 'OPEL 4 — Maximum Escalation'
}
opel_classes = {1: 'opel1', 2: 'opel2', 3: 'opel3', 4: 'opel4'}

pred_class = f"opel{int(prediction)}"
st.markdown(f'<div class="prediction-box {pred_class}">OPEL Level: {opel_names[int(prediction)]}</div>', unsafe_allow_html=True)

# ============== CONFIDENCE SCORES ==============
st.subheader("📊 Prediction Confidence")
col_conf1, col_conf2, col_conf3, col_conf4 = st.columns(4)

with col_conf1:
    st.markdown('<div class="metric-box"><strong>OPEL 1</strong><br>' + f'{probabilities[0]:.1%}' + '</div>', unsafe_allow_html=True)
with col_conf2:
    st.markdown('<div class="metric-box"><strong>OPEL 2</strong><br>' + f'{probabilities[1]:.1%}' + '</div>', unsafe_allow_html=True)
with col_conf3:
    st.markdown('<div class="metric-box"><strong>OPEL 3</strong><br>' + f'{probabilities[2]:.1%}' + '</div>', unsafe_allow_html=True)
with col_conf4:
    st.markdown('<div class="metric-box"><strong>OPEL 4</strong><br>' + f'{probabilities[3]:.1%}' + '</div>', unsafe_allow_html=True)

# ============== KEY DRIVERS ==============
st.header("🔍 Key Drivers of This Prediction")

# Simplified feature importance based on Session 3 findings
key_drivers = {
    'bed_occupancy_pct': {'value': bed_occ, 'importance': 0.40, 'direction': 'Higher occupancy increases OPEL risk'},
    'ae_4hr_breach_pct': {'value': ae_breach, 'importance': 0.22, 'direction': 'Higher breach rate increases OPEL risk'},
    'temperature_c': {'value': temp, 'importance': 0.13, 'direction': 'Lower temperature increases OPEL risk (winter effect)'},
}

for feature, data in key_drivers.items():
    col_name, col_val, col_imp = st.columns([2, 1, 1])
    with col_name:
        st.write(f"**{feature.replace('_', ' ').title()}**")
        st.caption(data['direction'])
    with col_val:
        st.metric("Value", f"{data['value']:.2f}")
    with col_imp:
        st.metric("Importance", f"{data['importance']:.0%}")

# ============== SAFETY NOTICE ==============
st.markdown('''
    <div class="info-box">
    <strong>⚠️ Safety & Governance Notice</strong><br><br>
    This model achieves <strong>92% accuracy</strong> but has a <strong>10% miss rate on OPEL 4</strong> (critical escalation).
    <br><br>
    <strong>This tool is for decision support only.</strong> Do not use for autonomous escalation. Always combine with:
    <ul>
    <li>Clinical judgment from operational team</li>
    <li>Real-time situational awareness</li>
    <li>Historical patterns and context</li>
    <li>Human review before any escalation action</li>
    </ul>
    For full model governance documentation, see the Clinical Safety & AI Governance Report.
    </div>
''', unsafe_allow_html=True)

# ============== FOOTER ==============
st.markdown("---")
st.markdown("""
**NHS OPEL Level Predictor v1.0** | Model trained on synthetic ICB data | 
[View Governance Report](https://github.com/your-repo) | 
[Model Card](https://github.com/your-repo) | 
Last updated: May 2026
""")
