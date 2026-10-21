"""Streamlit dashboard: Great Britain road safety (STATS19). Run: streamlit run app/streamlit_app.py"""
import json, pathlib
import pandas as pd, plotly.express as px, streamlit as st

P = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed"
st.set_page_config(page_title="GB Road Safety", layout="wide")

@st.cache_data
def load():
    k = pd.read_parquet(P / "casualties.parquet")
    return (k, pd.read_csv(P / "local_authority_rates.csv"), pd.read_csv(P / "hour_by_day.csv"),
            pd.read_csv(P / "ksi_locations.csv"), pd.read_csv(P / "model_metrics.csv", index_col=0),
            pd.read_csv(P / "model_feature_importance.csv"), json.loads((P / "key_findings.json").read_text()))
k, la, hw, loc, mm, imp, F = load()

st.title("Great Britain road safety, " + F["years"])
st.caption("DfT STATS19 reported road casualties (Open Government Licence) and ONS population estimates. "
           "Built by Anthony Adusei-Poku (github.com/anthony07-max).")
c = st.columns(4)
c[0].metric(f"Casualties {F['last_year']}", f"{F['casualties_last']:,}", f"{F['casualties_last']-F['casualties_first']:+,} vs {F['first_year']}", delta_color="inverse")
c[1].metric(f"Killed {F['last_year']}", f"{F['killed_last']:,}", f"{F['killed_last']-F['killed_first']:+,} vs {F['first_year']}", delta_color="inverse")
c[2].metric(f"KSI (adjusted) {F['last_year']}", f"{F['ksi_adjusted_last']:,}", f"{F['ksi_adjusted_last']-F['ksi_adjusted_first']:+,} vs {F['first_year']}", delta_color="inverse")
c[3].metric("Pedestrians, cyclists, motorcyclists", f"{F['vulnerable_share_ksi_pct']['London']}% of London KSI",
            f"{F['vulnerable_share_ksi_pct']['Rest of GB']}% in rest of GB", delta_color="off")

area = st.sidebar.radio("Area", ["Great Britain", "London", "Rest of GB"])
d = k if area == "Great Britain" else k[k.area == area]
t1, t2, t3, t4, t5 = st.tabs(["Trends & road users", "When collisions happen", "Local authorities", "Map", "Severity model"])
with t1:
    measure = st.selectbox("Measure", ["ksi_adjusted", "fatal", "ksi", "casualties"],
                           format_func={"ksi_adjusted": "KSI (severity-adjusted)", "fatal": "Killed", "ksi": "KSI (as reported)", "casualties": "All casualties"}.get)
    g = d.assign(casualties=1).groupby(["year", "road_user"], as_index=False)[measure].sum()
    st.plotly_chart(px.bar(g, x="year", y=measure, color="road_user", title=f"{area}: {measure} by road user"), width="stretch")
    a = d.assign(casualties=1).groupby("age_band", as_index=False)[["casualties", "ksi_adjusted"]].sum()
    st.plotly_chart(px.bar(a, x="age_band", y="ksi_adjusted", title="KSI casualties by age band (adjusted)"), width="stretch")
with t2:
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    h = d.groupby(["day_of_week_label", "hour"], as_index=False).ksi_adjusted.sum()
    piv = h.pivot(index="day_of_week_label", columns="hour", values="ksi_adjusted").reindex(days)
    st.plotly_chart(px.imshow(piv, aspect="auto", color_continuous_scale="Reds",
                              title=f"{area}: KSI casualties by day and hour"), width="stretch")
with t3:
    only_london = st.checkbox("London boroughs only", value=area == "London")
    v = la[la.london] if only_london else la
    v = v[v.avg_population >= st.slider("Minimum population", 0, 300000, 50000, 10000)]
    st.plotly_chart(px.scatter(v, x="avg_population", y="ksi_per_100k_yr", hover_name="la_name", color="london", log_x=True,
                               title="KSI casualties per 100,000 residents per year vs population"), width="stretch")
    st.dataframe(v[["la_name", "avg_population", "ksi_adjusted", "killed", "ksi_per_100k_yr", "casualties_per_100k_yr"]],
                 width="stretch", hide_index=True)
with t4:
    yrs = st.multiselect("Years", sorted(loc.year.unique()), default=[loc.year.max()])
    m = loc[loc.year.isin(yrs)]
    if area == "London": m = m[m.london]
    elif area == "Rest of GB": m = m[~m.london]
    st.caption(f"{len(m):,} fatal and serious collisions")
    st.plotly_chart(px.density_map(m, lat="latitude", lon="longitude", radius=4, zoom=9 if area == "London" else 5,
                                   center={"lat": 51.51, "lon": -0.12} if area == "London" else {"lat": 53.5, "lon": -2},
                                   map_style="open-street-map", height=650), width="stretch")
with t5:
    st.write(f"Predicting whether a collision is fatal or serious (base rate {F['model']['ksi_base_rate_test_pct']}%). "
             f"Trained on {F['model']['train_years']}, tested on {F['model']['test_year']}. Class weights balance the classes.")
    st.dataframe(mm, width="stretch")
    st.plotly_chart(px.bar(imp.sort_values("auc_drop"), x="auc_drop", y=imp.columns[0], orientation="h",
                           title="Permutation importance (drop in ROC AUC)"), width="stretch")
