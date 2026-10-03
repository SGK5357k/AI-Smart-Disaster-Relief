from pathlib import Path
import joblib
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; MODELS=ROOT/"models"
FEATURES=["disaster_type","rainfall_mm","temperature_c","humidity_pct","wind_speed_kmph","river_level_m","quake_magnitude","slope_instability_index","population_density","affected_population","population","population_density_area"]
model=joblib.load(MODELS/"severity_model.pkl"); scaler=joblib.load(MODELS/"scaler.pkl"); encoders=joblib.load(MODELS/"label_encoders.pkl")

def predict_severity(disaster_type,rainfall_mm,temperature_c,humidity_pct,wind_speed_kmph,river_level_m,quake_magnitude,slope_instability_index,population_density,affected_population,population,population_density_area):
    enc=encoders["disaster_type"]; value=int(enc.transform([disaster_type])[0])
    x=pd.DataFrame([[value,rainfall_mm,temperature_c,humidity_pct,wind_speed_kmph,river_level_m,quake_magnitude,slope_instability_index,population_density,affected_population,population,population_density_area]],columns=FEATURES)
    xs=scaler.transform(x); pred=int(model.predict(xs)[0]); severity=encoders["severity"].inverse_transform([pred])[0]; probs=model.predict_proba(xs)[0] if hasattr(model,"predict_proba") else None
    return severity,float(np.max(probs)) if probs is not None else None
