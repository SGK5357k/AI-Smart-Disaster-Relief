import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Dashboard",page_icon="📊",layout="wide")
ROOT=Path(__file__).resolve().parents[1]; df=pd.read_csv(ROOT/"datasets"/"raw"/"master_dataset.csv")
st.title("📊 Disaster Analytics Dashboard"); st.caption("Overview of the disaster and resource dataset."); st.divider()
a,b,c,d=st.columns(4); a.metric("🌍 Events",f"{len(df):,}"); b.metric("🗺 States",df.state.nunique()); c.metric("⚠ Avg Severity Score",f"{df.severity_score.mean():.2f}"); d.metric("👥 Affected Population",f"{int(df.affected_population.sum()):,}")
left,right=st.columns(2)
with left: st.plotly_chart(px.histogram(df,x="disaster_type",color="disaster_type",text_auto=True,title="Disaster Type Distribution"),use_container_width=True)
with right: st.plotly_chart(px.histogram(df,x="severity",color="severity",text_auto=True,title="Severity Distribution"),use_container_width=True)
res=pd.DataFrame({"Resource":["Food (kg)","Water (L)","Medicine Kits","Shelters","Ambulances","Rescue Teams"],"Average":[df.food_kg.mean(),df.water_liters.mean(),df.medicine_kits.mean(),df.shelters_needed.mean(),df.ambulances_needed.mean(),df.rescue_teams_needed.mean()]})
st.plotly_chart(px.bar(res,x="Resource",y="Average",color="Resource",text_auto=".1f",title="Average Resource Requirements"),use_container_width=True)
state=df.groupby("state",as_index=False).agg(Total_Events=("event_id","count"),Affected_Population=("affected_population","sum"),Avg_Severity=("severity_score","mean"))
left,right=st.columns(2)
with left: st.plotly_chart(px.bar(state,x="state",y="Total_Events",color="Total_Events",text_auto=True,title="Events by State"),use_container_width=True)
with right: st.plotly_chart(px.bar(state,x="state",y="Affected_Population",color="Affected_Population",title="Affected Population by State"),use_container_width=True)
st.subheader("🚨 Top 10 High-Severity Events"); cols=["event_id","state","disaster_type","severity","severity_score","affected_population","food_kg","water_liters"]; st.dataframe(df.nlargest(10,"severity_score")[cols],use_container_width=True,hide_index=True)
with st.expander("📋 Complete Dataset"): st.dataframe(df,use_container_width=True,hide_index=True)
st.download_button("⬇ Download Dataset",df.to_csv(index=False),"master_dataset.csv","text/csv",use_container_width=True)
st.divider(); st.caption("AI Smart Disaster Relief Resource Allocation System | Machine Learning Internship Project")
