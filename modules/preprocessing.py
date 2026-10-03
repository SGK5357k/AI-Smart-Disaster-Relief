from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE=Path(__file__).resolve().parent.parent; RAW=BASE/"datasets"/"raw"/"master_dataset.csv"; OUT=BASE/"datasets"/"processed"/"clean_dataset.csv"; ENC=BASE/"models"/"label_encoders.pkl"

def preprocess():
    df=pd.read_csv(RAW).drop_duplicates().copy()
    num=df.select_dtypes(include="number").columns; cat=df.select_dtypes(include=["object","string"]).columns
    df[num]=df[num].apply(lambda s:s.fillna(s.median()))
    df[cat]=df[cat].fillna("Unknown")
    df=df.drop(columns=[c for c in ["event_id","area_id"] if c in df.columns])
    encoders={}
    for col in cat:
        if col in df.columns:
            e=LabelEncoder(); df[col]=e.fit_transform(df[col]); encoders[col]=e
    OUT.parent.mkdir(parents=True,exist_ok=True); ENC.parent.mkdir(parents=True,exist_ok=True); df.to_csv(OUT,index=False); joblib.dump(encoders,ENC); return df,encoders
if __name__=="__main__":
    df,_=preprocess(); print(f"Saved {len(df)} rows to {OUT}")
