"""SQL analysis, severity model and charts. Run after clean.py.
Writes charts/*.png, data/processed/*.csv summary tables and data/processed/key_findings.json."""
import json, sqlite3, pathlib, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC, CH = ROOT / "data" / "processed", ROOT / "charts"
CH.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .3, "font.size": 10})
BLUE, DARK, RED, GREY = "#1D70B8", "#0B0C0C", "#D4351C", "#A0AEC0"

c = pd.read_parquet(PROC / "collisions.parquet")
k = pd.read_parquet(PROC / "casualties.parquet")
pop = pd.read_csv(PROC / "population_la.csv")
con = sqlite3.connect(":memory:")
c.drop(columns=["date"]).to_sql("collisions", con)
k.drop(columns=["date"]).to_sql("casualties", con)
pop.to_sql("population", con)
q = lambda sql: pd.read_sql(sql, con)
F = {"years": f"{c.year.min()}-{c.year.max()}", "n_collisions": len(c), "n_casualties": len(k)}

# 1. National trend -----------------------------------------------------------------
yr = q("""SELECT c.year, COUNT(*) AS casualties, SUM(fatal) AS killed, SUM(ksi) AS ksi_reported,
                 ROUND(SUM(ksi_adjusted)) AS ksi_adjusted, p.population
          FROM casualties c JOIN (SELECT year, SUM(population) AS population FROM population GROUP BY year) p USING (year)
          GROUP BY c.year ORDER BY c.year""")
yr["ksi_adj_per_100k"] = (yr.ksi_adjusted / yr.population * 1e5).round(1)
yr.to_csv(PROC / "gb_annual.csv", index=False)
first, last = yr.iloc[0], yr.iloc[-1]
F.update(first_year=int(first.year), last_year=int(last.year),
         casualties_first=int(first.casualties), casualties_last=int(last.casualties),
         killed_first=int(first.killed), killed_last=int(last.killed), killed_total=int(yr.killed.sum()),
         killed_peak_year=int(yr.loc[yr.killed.idxmax(), "year"]), killed_peak=int(yr.killed.max()),
         ksi_reported_first=int(first.ksi_reported), ksi_reported_last=int(last.ksi_reported),
         ksi_adjusted_first=int(first.ksi_adjusted), ksi_adjusted_last=int(last.ksi_adjusted),
         injury_based_share_first=round(c[c.year == c.year.min()].collision_injury_based.mean() * 100, 1),
         injury_based_share_last=round(c[c.year == c.year.max()].collision_injury_based.mean() * 100, 1))

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
axs[0].plot(yr.year, yr.ksi_reported / 1e3, "o-", color=GREY, label="KSI as reported")
axs[0].plot(yr.year, yr.ksi_adjusted / 1e3, "o-", color=BLUE, lw=2.2, label="KSI, DfT severity-adjusted")
axs[0].set(title="Killed or seriously injured casualties (thousands)", ylim=(0, None)); axs[0].legend(loc="lower right")
axs[1].bar(yr.year, yr.killed, color=RED)
for x, v in zip(yr.year, yr.killed): axs[1].text(x, v + 15, f"{v:,}", ha="center", fontsize=9)
axs[1].set(title="People killed on the roads", ylim=(0, yr.killed.max() * 1.15))
for a in axs: a.xaxis.set_major_locator(mt.MultipleLocator(1))
fig.suptitle(f"Reported road casualties, Great Britain {F['years']} (STATS19)")
fig.tight_layout(); fig.savefig(CH / "01_gb_trend.png"); plt.close(fig)

# 2. London vs rest of GB, per population --------------------------------------------
ar = q("""SELECT area, year, COUNT(*) AS casualties, SUM(fatal) AS killed, SUM(ksi_adjusted) AS ksi_adjusted
          FROM casualties GROUP BY area, year""")
pa = pop.assign(area=np.where(pop.la_code.str.startswith("E09"), "London", "Rest of GB")).groupby(["area", "year"], as_index=False).population.sum()
ar = ar.merge(pa, on=["area", "year"])
for m in ["casualties", "killed", "ksi_adjusted"]:
    ar[m + "_per_100k"] = (ar[m] / ar.population * 1e5).round(2)
ar.to_csv(PROC / "london_vs_rest.csv", index=False)
tot = ar.groupby("area")[["casualties", "killed", "ksi_adjusted", "population"]].sum()
for m in ["casualties", "killed", "ksi_adjusted"]:
    tot[m + "_per_100k"] = (tot[m] / tot.population * 1e5).round(2)
F["london_vs_rest_5yr_per_100k_per_year"] = tot[[x for x in tot.columns if x.endswith("per_100k")]].round(1).to_dict("index")
F["london_killed_total"] = int(tot.loc["London", "killed"])

share = q("""SELECT area, road_user, SUM(ksi_adjusted) AS ksi FROM casualties GROUP BY area, road_user""")
share["pct"] = share.ksi / share.groupby("area").ksi.transform("sum") * 100
share.to_csv(PROC / "ksi_share_by_road_user.csv", index=False)
sp = share.pivot(index="road_user", columns="area", values="pct").sort_values("London")
F["ksi_share_by_road_user_pct"] = sp.round(1).to_dict()
vru = ["Pedestrian", "Cyclist", "Motorcyclist"]
F["vulnerable_share_ksi_pct"] = sp.loc[vru].sum().round(1).to_dict()

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
w = .38; x = np.arange(len(sp))
axs[0].barh(x - w / 2, sp["Rest of GB"], w, color=GREY, label="Rest of GB")
axs[0].barh(x + w / 2, sp["London"], w, color=RED, label="London")
axs[0].set_yticks(x, sp.index); axs[0].xaxis.set_major_formatter(mt.PercentFormatter()); axs[0].legend(loc="lower right")
axs[0].set_title(f"Share of KSI casualties by road user, {F['years']}")
for a, col in [("Rest of GB", GREY), ("London", RED)]:
    d = ar[ar.area == a]; axs[1].plot(d.year, d.ksi_adjusted_per_100k, "o-", color=col, lw=2, label=a)
axs[1].set(title="KSI casualties per 100,000 residents (adjusted)", ylim=(0, None)); axs[1].legend()
axs[1].xaxis.set_major_locator(mt.MultipleLocator(1))
fig.tight_layout(); fig.savefig(CH / "02_london_vs_rest.png"); plt.close(fig)

# 3. Vulnerable road users trend ------------------------------------------------------
vt = q("""SELECT year, road_user, SUM(fatal) AS killed, SUM(ksi_adjusted) AS ksi_adjusted FROM casualties
          GROUP BY year, road_user""")
vt.to_csv(PROC / "road_user_trend.csv", index=False)
kp = vt.pivot(index="year", columns="road_user", values="killed")
F["killed_by_road_user_last_year"] = kp.iloc[-1].astype(int).sort_values(ascending=False).to_dict()
F["killed_by_road_user_first_year"] = kp.iloc[0].astype(int).sort_values(ascending=False).to_dict()
F["vulnerable_share_killed_pct_5yr"] = round(kp[vru].sum().sum() / kp.sum().sum() * 100, 1)
es = k[k.escooter_flag == 1].groupby("year").agg(casualties=("fatal", "size"), killed=("fatal", "sum"), ksi_adjusted=("ksi_adjusted", "sum"))
es.to_csv(PROC / "escooter_trend.csv")
F["escooter_casualties_by_year"] = es.casualties.astype(int).to_dict()
F["escooter_ksi_adjusted_by_year"] = es.ksi_adjusted.round().astype(int).to_dict()
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
cols = {"Pedestrian": RED, "Cyclist": "#00703C", "Motorcyclist": "#F47738", "Car occupant": BLUE}
for ru, col in cols.items():
    d = vt[vt.road_user == ru]
    axs[0].plot(d.year, d.ksi_adjusted / 1e3, "o-", color=col, label=ru)
    axs[1].plot(d.year, d.killed, "o-", color=col, label=ru)
axs[0].set(title="KSI casualties (adjusted, thousands)", ylim=(0, None)); axs[1].set(title="Killed", ylim=(0, None))
for a in axs: a.xaxis.set_major_locator(mt.MultipleLocator(1))
axs[1].legend(); fig.suptitle("Pedestrians, cyclists and motorcyclists vs car occupants, GB")
fig.tight_layout(); fig.savefig(CH / "03_road_users.png"); plt.close(fig)

# 4. Time of day x day of week ---------------------------------------------------------
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
hw = q("""SELECT day_of_week_label AS day, CAST(hour AS INT) AS hour, COUNT(*) AS collisions, SUM(ksi) AS ksi
          FROM collisions WHERE hour IS NOT NULL GROUP BY 1, 2""")
hw.to_csv(PROC / "hour_by_day.csv", index=False)
hm = hw.pivot(index="day", columns="hour", values="collisions").reindex(days) / c.year.nunique()
pk = hw.loc[hw.collisions.idxmax()]
F.update(peak_slot=f"{pk.day} {int(pk.hour):02d}:00-{int(pk.hour):02d}:59", peak_slot_collisions_per_year=round(pk.collisions / c.year.nunique()))
hr = hw.groupby("hour")[["collisions", "ksi"]].sum(); hr["ksi_pct"] = hr.ksi / hr.collisions * 100
F.update(ksi_pct_by_hour_max=[int(hr.ksi_pct.idxmax()), round(hr.ksi_pct.max(), 1)],
         ksi_pct_by_hour_min=[int(hr.ksi_pct.idxmin()), round(hr.ksi_pct.min(), 1)],
         ksi_pct_all=round(c.ksi.mean() * 100, 1),
         ksi_pct_0000_0559=round(hr.loc[0:5, "ksi"].sum() / hr.loc[0:5, "collisions"].sum() * 100, 1))
fig, axs = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
im = axs[0].imshow(hm.values, aspect="auto", cmap="Reds")
axs[0].set_yticks(range(7), days); axs[0].set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
axs[0].set(title="Reported injury collisions per year by hour and day, GB", xlabel="Hour of day"); axs[0].grid(False)
fig.colorbar(im, ax=axs[0], shrink=.8)
axs[1].bar(hr.index, hr.ksi_pct, color=[RED if h in range(0, 6) else BLUE for h in hr.index])
axs[1].axhline(F["ksi_pct_all"], color=DARK, ls="--", lw=1)
axs[1].set(title="% fatal or serious, by hour", xlabel="Hour of day")
axs[1].yaxis.set_major_formatter(mt.PercentFormatter())
fig.tight_layout(); fig.savefig(CH / "04_time_heatmap.png"); plt.close(fig)

# 5. Local authority rates per population ---------------------------------------------
la = q("""SELECT c.la_code, SUM(c.ksi_adjusted) AS ksi_adjusted, SUM(c.fatal) AS killed, COUNT(*) AS casualties
          FROM casualties c GROUP BY c.la_code""")
lp = pop.groupby(["la_code", "la_name"], as_index=False).population.sum()  # person-years 2021-2025
la = la.merge(lp, on="la_code")
la["ksi_per_100k_yr"] = (la.ksi_adjusted / la.population * 1e5).round(1)
la["casualties_per_100k_yr"] = (la.casualties / la.population * 1e5).round(1)
la["london"] = la.la_code.str.startswith("E09")
la["avg_population"] = (la.population / c.year.nunique()).round()
la = la.sort_values("ksi_per_100k_yr", ascending=False)
la.to_csv(PROC / "local_authority_rates.csv", index=False)
lon = la[la.london & (la.la_code != "E09000001")].sort_values("ksi_per_100k_yr")  # City of London excluded (tiny resident base)
F["london_borough_highest_ksi_rate"] = lon.iloc[-3:][::-1][["la_name", "ksi_per_100k_yr"]].values.tolist()
F["london_borough_lowest_ksi_rate"] = lon.iloc[:3][["la_name", "ksi_per_100k_yr"]].values.tolist()
F["newham_ksi_per_100k_yr"] = float(la.loc[la.la_name == "Newham", "ksi_per_100k_yr"].iloc[0])
big = la[la.avg_population >= 100000]
F["gb_la_highest_ksi_rate_pop100k_plus"] = big.iloc[:3][["la_name", "ksi_per_100k_yr"]].values.tolist()
F["gb_la_lowest_ksi_rate_pop100k_plus"] = big.iloc[-3:][::-1][["la_name", "ksi_per_100k_yr"]].values.tolist()
fig, ax = plt.subplots(figsize=(9, 9))
ax.barh(lon.la_name, lon.ksi_per_100k_yr, color=[RED if n == "Newham" else BLUE for n in lon.la_name])
gb_rate = F["london_vs_rest_5yr_per_100k_per_year"]["Rest of GB"]["ksi_adjusted_per_100k"]
ax.axvline(gb_rate, color=DARK, ls="--", lw=1, label=f"Rest of GB ({gb_rate})"); ax.legend(loc="lower right")
ax.set(title=f"London boroughs: KSI casualties per 100,000 residents per year, {F['years']}\n(severity-adjusted; City of London excluded)")
fig.tight_layout(); fig.savefig(CH / "05_london_boroughs.png"); plt.close(fig)

# 6. Maps (hexbin of KSI collision locations) ------------------------------------------
kc = c[(c.ksi == 1) & c.longitude.notna()]
fig, axs = plt.subplots(1, 2, figsize=(13, 8), gridspec_kw={"width_ratios": [1, 1.3]})
axs[0].hexbin(kc.longitude, kc.latitude, gridsize=110, bins="log", cmap="inferno_r", mincnt=1)
axs[0].set(title=f"Fatal and serious collisions, GB {F['years']}", xlim=(-8, 2), ylim=(49.8, 59)); axs[0].set_aspect(1.6)
lk = kc[kc.london]
hb = axs[1].hexbin(lk.longitude, lk.latitude, gridsize=70, cmap="inferno_r", mincnt=1)
axs[1].set(title=f"London: {len(lk):,} fatal and serious collisions"); axs[1].set_aspect(1.6)
fig.colorbar(hb, ax=axs[1], shrink=.6, label="Collisions per hexagon")
for a in axs: a.grid(False); a.set_xticks([]); a.set_yticks([])
fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(CH / "06_ksi_map.png"); plt.close(fig)
kc[["year", "latitude", "longitude", "collision_severity_label", "london"]].to_csv(PROC / "ksi_locations.csv", index=False)

print("Descriptive charts done")
# 7. Severity classification model ------------------------------------------------------
inv = k.assign(ped=k.road_user.eq("Pedestrian"), cyc=k.road_user.eq("Cyclist"), moto=k.road_user.eq("Motorcyclist"),
               old=k.age_of_casualty.ge(70)).groupby("collision_index")[["ped", "cyc", "moto", "old"]].max().astype(int)
m = c.merge(inv, left_on="collision_index", right_index=True, how="left").fillna({"ped": 0, "cyc": 0, "moto": 0, "old": 0})
cat = ["road_type_label", "first_road_class_label", "junction_detail_label", "light_conditions_label", "weather_conditions_label",
       "road_surface_conditions_label", "urban_or_rural_area_label", "day_of_week_label"]
num = ["speed_limit", "hour", "month", "number_of_vehicles", "number_of_casualties", "ped", "cyc", "moto", "old", "collision_injury_based"]
m[cat] = m[cat].fillna("Unknown").astype(str)
train, test = m[m.year < m.year.max()], m[m.year == m.year.max()]
Xtr, ytr, Xte, yte = train[cat + num], train.ksi, test[cat + num], test.ksi
pre_lr = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cat),
                            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num)])
lr = make_pipeline(pre_lr, LogisticRegression(class_weight="balanced", max_iter=500))
print("Fitting logistic regression ..."); lr.fit(Xtr, ytr)
pre_gb = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
                            ("num", "passthrough", num)])
gb = make_pipeline(pre_gb, HistGradientBoostingClassifier(class_weight="balanced", max_iter=300, learning_rate=.08,
                   categorical_features=list(range(len(cat))), random_state=42))
print("Fitting gradient boosting ..."); gb.fit(Xtr, ytr)
res = {}
for name, mod, X in [("Logistic regression", lr, Xte), ("Gradient boosting", gb, Xte)]:
    p = mod.predict_proba(X)[:, 1]; yhat = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, yhat).ravel()
    res[name] = dict(roc_auc=round(roc_auc_score(yte, p), 3), pr_auc=round(average_precision_score(yte, p), 3),
                     recall=round(recall_score(yte, yhat), 3), precision=round(precision_score(yte, yhat), 3),
                     tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), _p=p)
F["model"] = {"train_years": f"{train.year.min()}-{train.year.max()}", "test_year": int(test.year.max()),
              "n_train": len(train), "n_test": len(test), "ksi_base_rate_test_pct": round(yte.mean() * 100, 1),
              "results": {n: {kk: v for kk, v in r.items() if kk != "_p"} for n, r in res.items()}}
Xs = Xte.sample(20000, random_state=1)
pi = permutation_importance(gb, Xs, yte.loc[Xs.index],
                            scoring="roc_auc", n_repeats=5, random_state=0)
imp = pd.Series(pi.importances_mean, index=cat + num).sort_values()
imp.rename("auc_drop").to_csv(PROC / "model_feature_importance.csv")
F["model"]["top_features"] = imp[::-1].head(5).round(4).to_dict()
pd.DataFrame(F["model"]["results"]).T.to_csv(PROC / "model_metrics.csv")
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
for (n, r), col in zip(res.items(), [GREY, BLUE]):
    fpr, tpr, _ = roc_curve(yte, r["_p"]); axs[0].plot(fpr, tpr, color=col, label=f"{n} (AUC {r['roc_auc']})")
    pr, rc, _ = precision_recall_curve(yte, r["_p"]); axs[1].plot(rc, pr, color=col, label=f"{n} (AP {r['pr_auc']})")
axs[0].plot([0, 1], [0, 1], ":", color=DARK); axs[0].set(title=f"ROC, test year {F['model']['test_year']}", xlabel="False positive rate", ylabel="True positive rate"); axs[0].legend(loc="lower right")
axs[1].axhline(yte.mean(), ls=":", color=DARK, label=f"Base rate {yte.mean():.1%}")
axs[1].set(title="Precision-recall", xlabel="Recall", ylabel="Precision"); axs[1].legend()
lbl = {"ped": "Pedestrian involved", "moto": "Motorcyclist involved", "cyc": "Cyclist involved", "old": "Casualty aged 70+",
       "collision_injury_based": "Injury-based reporting force"}
axs[2].barh([lbl.get(i, i.replace("_label", "").replace("_", " ")) for i in imp.tail(10).index], imp.tail(10), color=BLUE)
axs[2].set(title="Permutation importance (drop in AUC)")
fig.tight_layout(); fig.savefig(CH / "07_severity_model.png"); plt.close(fig)

(PROC / "key_findings.json").write_text(json.dumps(F, indent=2, default=str))
print(json.dumps(F, indent=2, default=str))
