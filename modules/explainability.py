from pathlib import Path
import joblib, pandas as pd

BASE=Path(__file__).resolve().parent.parent; MODELS=BASE/"models"; REPORT=BASE/"reports"; DATA=BASE/"datasets"/"processed"/"clean_dataset.csv"
FEATURES=["disaster_type","rainfall_mm","temperature_c","humidity_pct","wind_speed_kmph","river_level_m","quake_magnitude","slope_instability_index","population_density","affected_population","population","population_density_area"]

def feature_importance():
    model=joblib.load(MODELS/"severity_model.pkl"); values=getattr(model,"feature_importances_",None)
    if values is None: raise ValueError("The saved severity model does not expose feature_importances_.")
    out=pd.DataFrame({"Feature":FEATURES,"Importance":values}).sort_values("Importance",ascending=False); REPORT.mkdir(exist_ok=True); out.to_csv(REPORT/"shap_feature_importance.csv",index=False); return out

def model_summary(): return {"features":FEATURES,"target":"severity","samples":len(pd.read_csv(DATA))}

if __name__=="__main__": print(feature_importance())
