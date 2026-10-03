import streamlit as st
st.set_page_config(page_title="Home",page_icon="🏠",layout="wide")
st.title("🏠 AI Smart Disaster Relief")
st.subheader("Intelligent Disaster Management using Machine Learning")
st.write("Use the navigation to predict severity, estimate resources, allocate available resources, inspect the map and review reports.")
st.divider()
a,b,c=st.columns(3)
a.info("⚠️ **Severity Prediction**\n\nPredict Low, Medium, High or Critical severity using the trained classifier.")
b.info("📦 **Resource Prediction**\n\nEstimate six emergency resource requirements.")
c.info("🚑 **Smart Allocation**\n\nFind a reachable relief center using open-road and stock constraints.")

st.caption("AI Smart Disaster Relief Resource Allocation System | Machine Learning Internship Project")
