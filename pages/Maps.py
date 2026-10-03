import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from pathlib import Path

st.set_page_config(page_title="Disaster Resource Map",page_icon="🗺️",layout="wide")
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/"datasets"/"raw"
@st.cache_data
def load(): return (pd.read_csv(RAW/"relief_centers_updated.csv"),pd.read_csv(RAW/"hospitals.csv"),pd.read_csv(RAW/"shelters.csv"))
relief,hospitals,shelters=load()

st.title("🗺️ Disaster Resource Map"); st.caption("Interactive locations of relief centers, hospitals and shelters."); st.divider()
a,b=st.columns(2)
with a: show_r=st.checkbox("Relief Centers",True); show_h=st.checkbox("Hospitals",True)
with b: show_s=st.checkbox("Shelters",True); zoom=st.slider("Zoom",4,15,6)
frames=[x[["latitude","longitude"]] for x in (relief,hospitals,shelters) if not x.empty]
coords=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame([[20.5937,78.9629]],columns=["latitude","longitude"])
center=[float(coords.latitude.mean()),float(coords.longitude.mean())]
m=folium.Map(location=center,zoom_start=zoom,control_scale=True)
if show_r:
    for _,r in relief.iterrows():
        folium.Marker([r.latitude,r.longitude],tooltip=f"Relief {r.center_id}",popup=f"<b>Relief Center</b><br>Center: {r.center_id}<br>Area: {r.area_id}<br>Food: {r.food_stock_kg:,.0f} kg<br>Water: {r.water_stock_liters:,.0f} L<br>Medicine: {r.medicine_stock_kits:,.0f}<br>Rescue: {r.rescue_teams_available}",icon=folium.Icon(color="green",icon="plus-sign")).add_to(m)
if show_h:
    for _,r in hospitals.iterrows(): folium.Marker([r.latitude,r.longitude],tooltip=f"Hospital {r.hospital_id}",popup=f"<b>Hospital</b><br>ID: {r.hospital_id}<br>Area: {r.area_id}<br>Beds: {r.bed_capacity}<br>Ambulances: {r.ambulances_available}",icon=folium.Icon(color="red",icon="plus")).add_to(m)
if show_s:
    for _,r in shelters.iterrows(): folium.Marker([r.latitude,r.longitude],tooltip=f"Shelter {r.shelter_id}",popup=f"<b>Shelter</b><br>ID: {r.shelter_id}<br>Area: {r.area_id}<br>Capacity: {r.capacity_people}",icon=folium.Icon(color="blue",icon="home")).add_to(m)
st_folium(m,height=620,returned_objects=[])
st.divider(); c1,c2,c3=st.columns(3); c1.metric("🚑 Relief Centers",len(relief)); c2.metric("🏥 Hospitals",len(hospitals)); c3.metric("🏠 Shelters",len(shelters))
with st.expander("View Relief Centers"): st.dataframe(relief,use_container_width=True,hide_index=True)
with st.expander("View Hospitals"): st.dataframe(hospitals,use_container_width=True,hide_index=True)
with st.expander("View Shelters"): st.dataframe(shelters,use_container_width=True,hide_index=True)
st.divider(); st.caption("AI Smart Disaster Relief Resource Allocation System | Machine Learning Internship Project")
