import streamlit as st
import pandas as pd
from pathlib import Path
from services.allocation_service import complete_allocation

st.set_page_config(page_title="Resource Allocation",page_icon="🚑",layout="wide")
ROOT=Path(__file__).resolve().parents[1]
POP=pd.read_csv(ROOT/"datasets"/"raw"/"population_data.csv")
AREAS=sorted(POP.area_id.unique())

st.title("🚑 Smart Resource Allocation")
st.caption("Find a reachable relief center using open-road distance and stock constraints.")
st.divider()
state=st.selectbox("State",sorted(POP.state.unique()))
state_areas=sorted(POP.loc[POP.state==state,"area_id"].unique())
area=st.selectbox("Affected Area",state_areas)

c1,c2=st.columns(2)
with c1:
    food=st.number_input("Food Required (kg)",0.0,10000000.0,1200.0)
    water=st.number_input("Water Required (L)",0.0,10000000.0,2500.0)
with c2:
    medicine=st.number_input("Medicine Kits",0.0,100000.0,180.0)
    rescue=st.number_input("Rescue Teams",0,1000,4)

if st.button("🚀 Allocate Resources",type="primary",use_container_width=True):
    try:
        r=complete_allocation(area,food,water,medicine,rescue)
        if r["status"]=="FAILED": st.error(r["message"])
        else:
            st.success(f"✅ Resources can be allocated from {r['center_id']} via an open route.")
            a,b,c=st.columns(3); a.metric("📍 Relief Center",r["center_id"]); b.metric("🛣 Distance",f"{r['distance_km']:.1f} km"); c.metric("🏠 Shelter",r["shelter_id"])
            st.subheader("📦 Allocation Details")
            details=pd.DataFrame([{"Resource":"Food","Requested":r["food_allocated"],"Center Stock":r["food_stock"]},{"Resource":"Water","Requested":r["water_allocated"],"Center Stock":r["water_stock"]},{"Resource":"Medicine","Requested":r["medicine_allocated"],"Center Stock":r["medicine_stock"]},{"Resource":"Rescue Teams","Requested":r["rescue_teams_allocated"],"Center Stock":r["rescue_available"]}])
            st.dataframe(details,use_container_width=True,hide_index=True)
            x,y=st.columns(2); x.info(f"🏥 Hospital: {r['hospital_id']} | Beds: {r['bed_capacity']:,} | Ambulances: {r['ambulances_available']}"); y.info(f"🏠 Shelter: {r['shelter_id']} | Capacity: {r['shelter_capacity']:,}")
            report=pd.DataFrame([r]); st.download_button("⬇ Download Allocation Report",report.to_csv(index=False),"allocation_report.csv","text/csv",use_container_width=True)
    except Exception as e: st.error(f"Allocation failed: {e}")
st.divider(); st.caption("AI Smart Disaster Relief Resource Allocation System | Machine Learning Internship Project")
