from pathlib import Path
import joblib, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

BASE=Path(__file__).resolve().parent.parent; DATA=BASE/"datasets"/"processed"/"clean_dataset.csv"; MODELS=BASE/"models"; REPORT=BASE/"reports"; MODELS.mkdir(exist_ok=True); REPORT.mkdir(exist_ok=True)
FEATURES=["disaster_type","rainfall_mm","temperature_c","humidity_pct","wind_speed_kmph","river_level_m","quake_magnitude","slope_instability_index","population_density","affected_population","population","population_density_area"]; TARGET="severity"

def train():
    df=pd.read_csv(DATA); X=df[FEATURES]; y=df[TARGET]; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y); scaler=StandardScaler(); Xtrs=scaler.fit_transform(Xtr); Xtes=scaler.transform(Xte); joblib.dump(scaler,MODELS/"scaler.pkl")
    models={"Decision Tree":DecisionTreeClassifier(random_state=42),"Random Forest":RandomForestClassifier(n_estimators=200,random_state=42),"XGBoost":XGBClassifier(random_state=42,eval_metric="mlogloss")}; results={}; best=None; best_acc=-1; best_pred=None
    for name,m in models.items():
        m.fit(Xtrs,ytr); pred=m.predict(Xtes); acc=accuracy_score(yte,pred); results[name]={"Accuracy":acc,"Precision":precision_score(yte,pred,average="weighted"),"Recall":recall_score(yte,pred,average="weighted"),"F1":f1_score(yte,pred,average="weighted")}
        if acc>best_acc: best_acc=acc; best=m; best_pred=pred
    joblib.dump(best,MODELS/"severity_model.pkl"); pd.DataFrame(results).T.to_csv(REPORT/"model_metrics.csv"); plt.figure(figsize=(6,5)); plt.imshow(confusion_matrix(yte,best_pred)); plt.title("Severity Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.colorbar(); plt.savefig(REPORT/"confusion_matrix.png",bbox_inches="tight"); plt.close(); return results
if __name__=="__main__": print(train())
