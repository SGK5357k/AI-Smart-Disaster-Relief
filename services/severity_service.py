from pathlib import Path
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
FEATURES = ["disaster_type","rainfall_mm","temperature_c","humidity_pct","wind_speed_kmph","river_level_m","quake_magnitude","slope_instability_index","population_density","affected_population","population","population_density_area"]

model = joblib.load(MODEL_DIR / "severity_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")

def _encode(encoder, value):
    values=list(encoder.classes_)
    if value not in values:
        raise ValueError(f"Unknown value '{value}'. Allowed values: {values}")
    return int(encoder.transform([value])[0])

def predict_severity(disaster_type, rainfall_mm, temperature_c, humidity_pct, wind_speed_kmph, river_level_m, quake_magnitude, slope_instability_index, population_density, affected_population, population, population_density_area):
    row={"disaster_type":_encode(encoders["disaster_type"], disaster_type),"rainfall_mm":float(rainfall_mm),"temperature_c":float(temperature_c),"humidity_pct":float(humidity_pct),"wind_speed_kmph":float(wind_speed_kmph),"river_level_m":float(river_level_m),"quake_magnitude":float(quake_magnitude),"slope_instability_index":float(slope_instability_index),"population_density":float(population_density),"affected_population":float(affected_population),"population":float(population),"population_density_area":float(population_density_area)}
    x=pd.DataFrame([row],columns=FEATURES)
    xs=scaler.transform(x)
    pred=int(model.predict(xs)[0])
    severity=encoders["severity"].inverse_transform([pred])[0]
    proba=model.predict_proba(xs)[0] if hasattr(model,"predict_proba") else None
    confidence=float(np.max(proba)) if proba is not None else None
    class_probs={encoders["severity"].inverse_transform([i])[0]:float(p) for i,p in enumerate(proba)} if proba is not None else {}
    return {"prediction":pred,"severity":severity,"confidence":confidence,"class_probabilities":class_probs}
