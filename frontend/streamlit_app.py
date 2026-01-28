"""
Streamlit Dashboard for GreenRoute Analytics
"""

import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="GreenRoute", layout="wide")
st.title("🌱 GreenRoute Analytics")

API_URL = "http://localhost:5000/api"

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Focus", "Efficiency")
with col2:
    st.metric("Goal", "30-40% CO2 ↓")
with col3:
    st.metric("Benefit", "20-30% Cost ↓")

st.divider()

# ============ PILLAR 1 ============
st.header("1️⃣ Operational Inefficiencies")

try:
    response = requests.get(f"{API_URL}/inefficiencies")
    data = response.json()
    ineff = data['inefficiencies']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("❌ Inefficient Routes", f"{ineff['total_inefficient']:,}", f"{ineff['percentage']:.1f}%")
        st.metric("💸 Wasted Cost/Month", f"₹{ineff['wasted_cost_total']:,.0f}")
    with col2:
        st.metric("🔥 Extra CO2/Month", f"{ineff['wasted_co2_total_kg']:,.0f} kg")
        st.metric("💚 Reduceable (25%)", f"{ineff['reduceable_co2_25pct_kg']:,.0f} kg/month")
    
    for rec in data['recommendations']:
        st.success(f"**{rec['action']}** - {rec['impact_cost']} + {rec['impact_co2']}")

except:
    st.error("❌ Cannot connect to API. Make sure it's running: python backend/app.py")

st.divider()

# ============ PILLAR 2 ============
st.header("2️⃣ Best Delivery Method")

try:
    response = requests.get(f"{API_URL}/last-mile-comparison")
    rec = response.json()
    
    st.success(f"**Winner: {rec['primary_method']}**")
    st.metric("CO2 Reduction", f"{rec['co2_reduction_percent']:.0f}%")
    
    st.subheader("Full Ranking")
    for item in rec['full_ranking']:
        st.info(f"{item['rank']}. {item['method']}: Score {item['score']:.3f}")

except:
    st.error("Cannot load recommendation")

st.divider()

# ============ PILLAR 3 ============
st.header("3️⃣ Disruption Prevention")

try:
    response = requests.get(f"{API_URL}/disruption-forecast")
    data = response.json()
    
    for action in data['actions']:
        st.warning(f"**{action['disruption_type']}** [{action['priority']}]\n\n{action['action']}")

except:
    st.error("Cannot load disruptions")

st.divider()
st.markdown("**GreenRoute Analytics** | Operational Efficiency → Environmental Impact")