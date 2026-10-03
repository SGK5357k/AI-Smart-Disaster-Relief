import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Reports",page_icon="📄",layout="wide")
ROOT=Path(__file__).resolve().parents[1]; REPORT=ROOT/"reports"

def read(name,cols=None):
    p=REPORT/name
    if not p.exists(): return pd.DataFrame(columns=cols or [])
    return pd.read_csv(p)
model=read("model_metrics.csv"); resource=read("resource_model_metrics.csv"); allocation=read("allocation_report.csv"); sample=read("sample_resource_prediction.csv")

st.title("📄 Reports & Analytics"); st.caption("Model performance, resource prediction and allocation reports."); st.divider()
a,b,c,d=st.columns(4); a.metric("🤖 Classifier Models",len(model)); b.metric("📦 Resource Metrics",len(resource)); c.metric("🚑 Allocation Records",len(allocation)); d.metric("🔮 Sample Predictions",len(sample))

st.subheader("🤖 Severity Model Performance")
st.dataframe(model,use_container_width=True,hide_index=False)
if not model.empty and "Accuracy" in model.columns:
    fig=px.bar(model.reset_index(names="Model"),x="Model",y="Accuracy",text_auto=".3f"); fig.update_layout(height=400); st.plotly_chart(fig,use_container_width=True)

st.subheader("📦 Resource Model Metrics")
st.dataframe(resource,use_container_width=True,hide_index=True)
if not resource.empty and set(["Metric","Value"]).issubset(resource.columns):
    fig=px.bar(resource,x="Metric",y="Value",text_auto=".3f"); fig.update_layout(height=400); st.plotly_chart(fig,use_container_width=True)

st.subheader("🚑 Allocation Report")
st.dataframe(allocation,use_container_width=True,hide_index=True)
if not allocation.empty and "distance_km" in allocation.columns:
    fig=px.bar(allocation.reset_index(),x="index",y="distance_km",text_auto=".1f"); fig.update_layout(height=350,xaxis_title="Record",yaxis_title="Distance (km)"); st.plotly_chart(fig,use_container_width=True)

st.subheader("📦 Sample Resource Prediction")
st.dataframe(sample,use_container_width=True,hide_index=True)
if not sample.empty:
    long=sample.melt(var_name="Resource",value_name="Quantity")
    fig=px.bar(long,x="Resource",y="Quantity",text_auto=".1f"); fig.update_layout(height=450); st.plotly_chart(fig,use_container_width=True)

st.divider(); st.subheader("⬇ Downloads")
for name,df,label in [("model_metrics.csv",model,"Model Metrics"),("resource_model_metrics.csv",resource,"Resource Metrics"),("allocation_report.csv",allocation,"Allocation Report"),("sample_resource_prediction.csv",sample,"Sample Predictions")]:
    st.download_button(f"📥 {label}",df.to_csv(index=False),name,"text/csv",use_container_width=True)
st.divider(); st.caption("AI Smart Disaster Relief Resource Allocation System | Machine Learning Internship Project")
