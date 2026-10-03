import streamlit as st
import pandas as pd
from pathlib import Path
from services.resource_service import predict_resources

st.set_page_config(page_title="Resource Prediction",page_icon="📦",layout="wide")
ROOT=Path(__file__).resolve().parents[1]
master=pd.read_csv(ROOT/"datasets"/"raw"/"master_dataset.csv")
pop=pd.read_csv(ROOT/"datasets"/"raw"/"population_data.csv")

st.title("📦 Resource Requirement Prediction")
st.caption("Predicts six emergency resource requirements using the saved multi-output regression model.")
st.divider()
state=st.selectbox("State",sorted(master.state.unique()))
sp=pop[pop.state==state]
def avg(col,fallback):
    s=sp[col].dropna() if col in sp else pd.Series(dtype=float)
    return float(s.mean()) if not s.empty else fallback

c1,c2,c3=st.columns(3)
with c1:
    disaster=st.selectbox("Disaster Type",sorted(master.disaster_type.unique()))
    severity=st.selectbox("Severity",sorted(master.severity.unique()))
    rainfall=st.number_input("Rainfall (mm)",0.0,2000.0,120.0)
    river=st.number_input("River Level (m)",0.0,30.0,4.5)
with c2:
    temperature=st.number_input("Temperature (°C)",-20.0,60.0,30.0)
    humidity=st.slider("Humidity (%)",0,100,75)
    wind=st.number_input("Wind Speed (km/h)",0.0,300.0,25.0)
    quake=st.number_input("Earthquake Magnitude",0.0,10.0,0.0)
with c3:
    slope=st.slider("Slope Instability",0.0,10.0,3.5)
    density=st.number_input("Population Density",0.0,100000.0,avg("population_density",1200.0))
    affected=st.number_input("Affected Population",0,2000000,5000)
    population=st.number_input("Population",0,5000000,int(avg("population",150000)))
    density_area=st.number_input("Area Population Density",0.0,100000.0,avg("population_density",1200.0))

if st.button("🚀 Predict Resources",type="primary",use_container_width=True):
    try:
        r=predict_resources(disaster,rainfall,temperature,humidity,wind,river,quake,slope,density,affected,severity,population,density_area)
        labels={"food_kg":"🍚 Food (kg)","water_liters":"💧 Water (L)","medicine_kits":"💊 Medicine Kits","shelters_needed":"🏠 Shelters","ambulances_needed":"🚑 Ambulances","rescue_teams_needed":"👨‍🚒 Rescue Teams"}
        st.success("✅ Resource prediction completed")
        cols=st.columns(3)
        items=list(labels.items())
        for i,(key,label) in enumerate(items): cols[i%3].metric(label,f"{round(r[key]):,}")
        chart=pd.DataFrame({"Resource":[labels[k] for k in labels],"Quantity":[r[k] for k in labels]})
        st.bar_chart(chart.set_index("Resource"))
        st.dataframe(chart.assign(Quantity=chart.Quantity.round(0).astype(int)),use_container_width=True,hide_index=True)
        report=pd.DataFrame([{"State":state,"Disaster":disaster,"Severity":severity,"Affected Population":affected,**{labels[k]:round(r[k]) for k in labels}}])
        st.download_button("⬇ Download Prediction Report",report.to_csv(index=False),"resource_prediction.csv","text/csv",use_container_width=True)
        st.info(f"Deploy approximately {round(r['rescue_teams_needed'])} rescue teams, {round(r['ambulances_needed'])} ambulances and prepare {round(r['shelters_needed'])} shelters based on the model output.")
    except Exception as e: st.error(f"Prediction failed: {e}")
st.divider(); st.caption("AI Smart Disaster Relief Resource Allocation System | Machine Learning Internship Project")
