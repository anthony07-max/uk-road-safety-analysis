"""Tidy the STATS19 collision and casualty files into analysis-ready tables in data/processed/.

Outputs
- collisions.parquet : one row per reported injury collision (2021-2025) with decoded labels
- casualties.parquet : one row per casualty, joined to collision time and place
- population_la.csv  : ONS mid-year population by local authority district and year
"""
import pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

guide = pd.read_excel(RAW / "data_guide.xlsx", sheet_name=0)
guide.columns = ["table", "field", "code", "label", "note"]

def labels(field, table="collision"):
    g = guide[(guide.field == field) & (guide.table == table)].dropna(subset=["code"])
    g = g[pd.to_numeric(g.code, errors="coerce").notna()]
    return dict(zip(g.code.astype(int), g.label.astype(str).str.strip()))

def road_user(t):
    if t == 0: return "Pedestrian"
    if t == 1: return "Cyclist"
    if t in (2, 3, 4, 5, 97): return "Motorcyclist"
    if t in (8, 9): return "Car occupant"
    if t in (19, 20, 21, 98): return "Goods vehicle occupant"
    if t in (10, 11, 18): return "Bus, coach or tram occupant"
    return "Other (incl. e-scooter, horse rider)"

# Collisions --------------------------------------------------------------------
c = pd.read_csv(RAW / "collisions.csv", low_memory=False, na_values=["-1", -1])
c["date"] = pd.to_datetime(c["date"], format="%d/%m/%Y")
c["hour"] = pd.to_numeric(c["time"].str[:2], errors="coerce")
c["month"] = c["date"].dt.month
c["year"] = c["collision_year"].astype(int)
# Harmonise to April 2023 local authority boundaries (the ONS population geography):
# Cumbria, North Yorkshire and Somerset districts merged into unitary authorities in 2023,
# Barnsley/Sheffield received new codes, and DfT codes Heathrow airport separately (it is in Hillingdon).
LA_2023 = {**{x: "E06000063" for x in ["E07000026", "E07000028", "E07000029"]},          # Cumberland
           **{x: "E06000064" for x in ["E07000027", "E07000030", "E07000031"]},          # Westmorland and Furness
           **{f"E070001{i}": "E06000065" for i in range(63, 70)},                         # North Yorkshire
           **{x: "E06000066" for x in ["E07000187", "E07000188", "E07000189", "E07000246"]},  # Somerset
           "E08000038": "E08000016", "E08000039": "E08000019", "EHEATHROW": "E09000017"}
c["la_code"] = c["local_authority_ons_district"].replace(LA_2023)
c["country"] = c["la_code"].str[0].map({"E": "England", "W": "Wales", "S": "Scotland"})
c["london"] = c["la_code"].str.startswith("E09", na=False)
c["area"] = np.where(c["london"], "London", "Rest of GB")
for f in ["collision_severity", "day_of_week", "light_conditions", "weather_conditions", "road_surface_conditions",
          "urban_or_rural_area", "road_type", "junction_detail", "first_road_class", "police_force"]:
    c[f + "_label"] = c[f].map(labels(f))
c["ksi"] = c["collision_severity"].isin([1, 2]).astype(int)
keep = ["collision_index", "year", "date", "month", "hour", "day_of_week", "day_of_week_label", "la_code", "country",
        "london", "area", "police_force_label", "longitude", "latitude", "collision_severity", "collision_severity_label", "ksi",
        "collision_injury_based", "collision_adjusted_severity_serious", "collision_adjusted_severity_slight",
        "number_of_vehicles", "number_of_casualties", "speed_limit", "road_type_label", "first_road_class_label",
        "junction_detail_label", "light_conditions_label", "weather_conditions_label", "road_surface_conditions_label",
        "urban_or_rural_area_label"]
c = c[keep]

# Casualties --------------------------------------------------------------------
k = pd.read_csv(RAW / "casualties.csv", low_memory=False)
k["road_user"] = k["casualty_type"].map(road_user)
k["severity_label"] = k["casualty_severity"].map(labels("casualty_severity", "casualty"))
k["age_band"] = k["age_band_of_casualty"].map(labels("age_band_of_casualty", "casualty"))
k["sex"] = k["sex_of_casualty"].map({1: "Male", 2: "Female"})
k["fatal"] = (k["casualty_severity"] == 1).astype(int)
k["ksi"] = k["casualty_severity"].isin([1, 2]).astype(int)
# DfT severity adjustment: expected KSI if every force used injury-based (CRASH) reporting
k["ksi_adjusted"] = k["fatal"] + k["casualty_adjusted_severity_serious"].clip(lower=0)
k = k.merge(c[["collision_index", "year", "date", "hour", "day_of_week_label", "la_code", "country", "london", "area"]],
            on="collision_index", how="left")
k = k[["collision_index", "year", "date", "hour", "day_of_week_label", "la_code", "country", "london", "area",
       "casualty_class", "casualty_type", "road_user", "escooter_flag", "casualty_severity", "severity_label",
       "fatal", "ksi", "ksi_adjusted", "age_of_casualty", "age_band", "sex", "casualty_imd_decile"]]

# Population (ONS mid-year estimates via Nomis) ----------------------------------
p = pd.read_csv(RAW / "population_la.csv").rename(columns=str.lower)
p = p.rename(columns={"geography_code": "la_code", "geography_name": "la_name", "date_name": "year", "obs_value": "population"})
p = p[p.la_code.str[0].isin(["E", "W", "S"])]

c.to_parquet(PROC / "collisions.parquet", index=False)
k.to_parquet(PROC / "casualties.parquet", index=False)
p.to_csv(PROC / "population_la.csv", index=False)
print(f"collisions {len(c):,}  casualties {len(k):,}  LA-years {len(p):,}")
print("collisions with no LA code:", c.la_code.isna().sum(), " LA codes not in ONS table:",
      sorted(set(c.la_code.dropna()) - set(p.la_code)))
