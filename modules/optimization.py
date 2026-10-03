from pathlib import Path
import pandas as pd

BASE=Path(__file__).resolve().parent.parent; RAW=BASE/"datasets"/"raw"; REPORT=BASE/"reports"; REPORT.mkdir(exist_ok=True)
relief=pd.read_csv(RAW/"relief_centers.csv"); roads=pd.read_csv(RAW/"roads.csv"); hospitals=pd.read_csv(RAW/"hospitals.csv"); shelters=pd.read_csv(RAW/"shelters.csv")

def find_best_relief_center(area_id):
    r=roads[(roads.area_id==area_id)&(roads.road_status.astype(str).str.lower()=="open")].sort_values("distance_km")
    for _,row in r.iterrows():
        c=relief[relief.center_id==row.center_id]
        if not c.empty:
            out=c.iloc[0].copy(); out["distance_km"]=row.distance_km; return out
    return None

def complete_allocation(area_id,food,water,medicine,rescue):
    c=find_best_relief_center(area_id)
    if c is None:return {"status":"FAILED","message":"No reachable relief center found."}
    if c.food_stock_kg<food:return {"status":"FAILED","message":"Insufficient Food Stock"}
    if c.water_stock_liters<water:return {"status":"FAILED","message":"Insufficient Water Stock"}
    if c.medicine_stock_kits<medicine:return {"status":"FAILED","message":"Insufficient Medicine Stock"}
    if c.rescue_teams_available<rescue:return {"status":"FAILED","message":"Insufficient Rescue Teams"}
    h=hospitals[hospitals.area_id==area_id].sort_values("ambulances_available",ascending=False); s=shelters[shelters.area_id==area_id].sort_values("capacity_people",ascending=False)
    return {"status":"SUCCESS","center_id":c.center_id,"area_id":area_id,"distance_km":float(c.distance_km),"food_allocated":food,"water_allocated":water,"medicine_allocated":medicine,"rescue_teams_allocated":rescue,"hospital_id":h.iloc[0].hospital_id if not h.empty else "Not Available","shelter_id":s.iloc[0].shelter_id if not s.empty else "Not Available","shelter_capacity":int(s.iloc[0].capacity_people) if not s.empty else 0}

def save_allocation_report(result): pd.DataFrame([result]).to_csv(REPORT/"allocation_report.csv",index=False)
