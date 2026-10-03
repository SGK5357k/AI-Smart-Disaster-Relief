from pathlib import Path
import joblib
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/"models"
FEATURES=["disaster_type","rainfall_mm","temperature_c","humidity_pct","wind_speed_kmph","river_level_m","quake_magnitude","slope_instability_index","population_density","affected_population","severity","population","population_density_area"]
TARGETS=["food_kg","water_liters","medicine_kits","shelters_needed","ambulances_needed","rescue_teams_needed"]
model=joblib.load(MODEL_DIR/"resource_model.pkl")
scaler=joblib.load(MODEL_DIR/"resource_scaler.pkl")
encoders=joblib.load(MODEL_DIR/"label_encoders.pkl")

def _encode(name,value):
    enc=encoders[name]
    if value not in list(enc.classes_):
        raise ValueError(f"Unknown {name}: {value}")
    return int(enc.transform([value])[0])

def predict_resources(disaster_type,rainfall_mm,temperature_c,humidity_pct,wind_speed_kmph,river_level_m,quake_magnitude,slope_instability_index,population_density,affected_population,severity,population,population_density_area):
    row={"disaster_type":_encode("disaster_type",disaster_type),"rainfall_mm":float(rainfall_mm),"temperature_c":float(temperature_c),"humidity_pct":float(humidity_pct),"wind_speed_kmph":float(wind_speed_kmph),"river_level_m":float(river_level_m),"quake_magnitude":float(quake_magnitude),"slope_instability_index":float(slope_instability_index),"population_density":float(population_density),"affected_population":float(affected_population),"severity":_encode("severity",severity),"population":float(population),"population_density_area":float(population_density_area)}
    x=pd.DataFrame([row],columns=FEATURES)
    pred=model.predict(scaler.transform(x))[0]
    return {target:max(0,float(value)) for target,value in zip(TARGETS,pred)}
