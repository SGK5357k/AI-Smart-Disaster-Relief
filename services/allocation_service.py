from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"datasets"/"raw"
RELIEF=pd.read_csv(RAW/"relief_centers_updated.csv")
ROADS=pd.read_csv(RAW/"roads.csv")
HOSPITALS=pd.read_csv(RAW/"hospitals.csv")
SHELTERS=pd.read_csv(RAW/"shelters.csv")

def allocate_resources(area_id,food_required,water_required,medicine_required,rescue_required):
    roads=ROADS[(ROADS.area_id==area_id)&(ROADS.road_status.astype(str).str.lower()=="open")].sort_values("distance_km")
    if roads.empty:
        return {"status":"FAILED","message":f"No open road found for {area_id}."}
    for _,road in roads.iterrows():
        centers=RELIEF[RELIEF.center_id==road.center_id]
        if centers.empty: continue
        center=centers.iloc[0]
        if center.food_stock_kg < food_required or center.water_stock_liters < water_required or center.medicine_stock_kits < medicine_required or center.rescue_teams_available < rescue_required:
            continue
        return {"status":"SUCCESS","center_id":str(center.center_id),"area_id":str(area_id),"distance_km":float(road.distance_km),"food_allocated":float(food_required),"water_allocated":float(water_required),"medicine_allocated":float(medicine_required),"rescue_teams_allocated":int(rescue_required),"food_stock":float(center.food_stock_kg),"water_stock":float(center.water_stock_liters),"medicine_stock":float(center.medicine_stock_kits),"rescue_available":int(center.rescue_teams_available),"latitude":float(center.latitude),"longitude":float(center.longitude)}
    return {"status":"FAILED","message":"No reachable relief center has enough stock for the requested resources."}

def complete_allocation(area_id,food,water,medicine,rescue):
    result=allocate_resources(area_id,food,water,medicine,rescue)
    if result["status"]=="FAILED": return result
    hospitals=HOSPITALS[HOSPITALS.area_id==area_id].sort_values(["ambulances_available","bed_capacity"],ascending=False)
    shelters=SHELTERS[SHELTERS.area_id==area_id].sort_values("capacity_people",ascending=False)
    result["hospital_id"]=str(hospitals.iloc[0].hospital_id) if not hospitals.empty else "Not Available"
    result["ambulances_available"]=int(hospitals.iloc[0].ambulances_available) if not hospitals.empty else 0
    result["bed_capacity"]=int(hospitals.iloc[0].bed_capacity) if not hospitals.empty else 0
    result["shelter_id"]=str(shelters.iloc[0].shelter_id) if not shelters.empty else "Not Available"
    result["shelter_capacity"]=int(shelters.iloc[0].capacity_people) if not shelters.empty else 0
    return result
