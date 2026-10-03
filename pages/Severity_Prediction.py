import streamlit as st
import pandas as pd
from pathlib import Path
from services.severity_service import predict_severity

st.set_page_config(page_title="Severity Prediction",page_icon="⚠️",layout="wide")
ROOT=Path(__file__).resolve().parents[1]
master=pd.read_csv(ROOT/"datasets"/"raw"/"master_dataset.csv")
pop=pd.read_csv(ROOT/"datasets"/"raw"/"population_data.csv")

st.title("⚠️ Disaster Severity Prediction")
st.caption("Uses the saved severity model with the exact 12 features used during training.")
st.divider()

states=sorted(master.state.dropna().unique())
state=st.selectbox("State",states)
state_pop=pop[pop.state==state]
def default(col, fallback):
    s=state_pop[col].dropna() if col in state_pop else pd.Series(dtype=float)
    return float(s.mean()) if not s.empty else fallback

c1,c2,c3=st.columns(3)
with c1:
    disaster=st.selectbox("Disaster Type",sorted(master.disaster_type.unique()))
    rainfall=st.number_input("Rainfall (mm)",0.0,2000.0,120.0)
    temperature=st.number_input("Temperature (°C)",-20.0,60.0,30.0)
    humidity=st.slider("Humidity (%)",0,100,75)
with c2:
    wind=st.number_input("Wind Speed (km/h)",0.0,300.0,25.0)
    river=st.number_input("River Level (m)",0.0,30.0,4.5)
    quake=st.number_input("Earthquake Magnitude",0.0,10.0,0.0)
    slope=st.slider("Slope Instability",0.0,10.0,3.5)
with c3:
    density=st.number_input("Population Density",0.0,100000.0,default("population_density",1200.0))
    affected=st.number_input("Affected Population",0,2000000,5000)
    population=st.number_input("Population",0,5000000,int(default("population",150000)))
    density_area=st.number_input("Area Population Density",0.0,100000.0,default("population_density",1200.0))

if st.button("🚀 Predict Severity",type="primary",use_container_width=True):
    try:
        result=predict_severity(disaster,rainfall,temperature,humidity,wind,river,quake,slope,density,affected,population,density_area)
        severity=result["severity"]; confidence=result["confidence"]
        st.subheader("📊 Prediction Result")
        a,b,c=st.columns(3); a.metric("⚠ Severity",severity); b.metric("🎯 Confidence",f"{confidence*100:.2f}%" if confidence is not None else "N/A"); c.metric("👥 Affected Population",f"{affected:,}")
        if severity=="Critical": st.error("🔴 CRITICAL RISK — immediate emergency coordination is required.")
        elif severity=="High": st.error("🟠 HIGH RISK — activate emergency response and prepare evacuation support.")
        elif severity=="Medium": st.warning("🟡 MEDIUM RISK — increase monitoring and prepare emergency resources.")
        else: st.success("🟢 LOW RISK — continue monitoring conditions.")
        if result["class_probabilities"]:
            probs=pd.DataFrame({"Severity":list(result["class_probabilities"]),"Probability":[v*100 for v in result["class_probabilities"].values()]})
            st.bar_chart(probs.set_index("Severity"))
        summary=pd.DataFrame({"Parameter":["State","Disaster","Rainfall","Temperature","Humidity","Wind Speed","River Level","Earthquake","Slope","Population Density","Affected Population","Population","Area Population Density"],"Value":[state,disaster,rainfall,temperature,humidity,wind,river,quake,slope,density,affected,population,density_area]})
        st.dataframe(summary,use_container_width=True,hide_index=True)
        report=pd.DataFrame([{"State":state,"Disaster":disaster,"Severity":severity,"Confidence":round(confidence*100,2) if confidence is not None else None,"Affected Population":affected}])
        st.download_button("⬇ Download Prediction Report",report.to_csv(index=False),"severity_prediction.csv","text/csv",use_container_width=True)
    except Exception as e: st.error(f"Prediction failed: {e}")
st.divider(); st.caption("AI Smart Disaster Relief Resource Allocation System | Machine Learning Internship Project")
