from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

BASE=Path(__file__).resolve().parent.parent; DATA=BASE/"datasets"/"processed"/"clean_dataset.csv"; MODELS=BASE/"models"; REPORT=BASE/"reports"; MODELS.mkdir(exist_ok=True); REPORT.mkdir(exist_ok=True)
FEATURES=["disaster_type","rainfall_mm","temperature_c","humidity_pct","wind_speed_kmph","river_level_m","quake_magnitude","slope_instability_index","population_density","affected_population","severity","population","population_density_area"]; TARGETS=["food_kg","water_liters","medicine_kits","shelters_needed","ambulances_needed","rescue_teams_needed"]

def train():
    df=pd.read_csv(DATA); X=df[FEATURES]; y=df[TARGETS]; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42); scaler=StandardScaler(); Xtrs=scaler.fit_transform(Xtr); Xtes=scaler.transform(Xte); joblib.dump(scaler,MODELS/"resource_scaler.pkl"); model=MultiOutputRegressor(RandomForestRegressor(n_estimators=300,max_depth=20,min_samples_split=5,min_samples_leaf=2,random_state=42,n_jobs=-1)); model.fit(Xtrs,ytr); pred=model.predict(Xtes); mae=mean_absolute_error(yte,pred); rmse=mean_squared_error(yte,pred)**.5; r2=r2_score(yte,pred,multioutput="uniform_average"); joblib.dump(model,MODELS/"resource_model.pkl"); pd.DataFrame({"Metric":["MAE","RMSE","R2"],"Value":[mae,rmse,r2]}).to_csv(REPORT/"resource_model_metrics.csv",index=False); imp=np.mean([e.feature_importances_ for e in model.estimators_],axis=0); pd.DataFrame({"Feature":FEATURES,"Importance":imp}).sort_values("Importance",ascending=False).to_csv(REPORT/"resource_feature_importance.csv",index=False); return mae,rmse,r2
if __name__=="__main__": print(train())
