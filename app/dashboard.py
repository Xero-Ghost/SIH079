"""
Standalone Streamlit dashboard for Streamlit Cloud deployment.
No separate FastAPI backend needed — model runs directly inside the app.

    streamlit run app/dashboard.py
"""
import json, sys
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bustcast.utils import load_cfg, P, get_device
from bustcast.dataset import split_feature_cols
from bustcast.model import BustCastModel
from bustcast.geo import norm_coords

st.set_page_config(page_title="BustCast — Forecast Bust Detection", layout="wide")

st.title("AI-Based Forecast Bust Detection — Medium-Range Weather Forecasts")
st.caption("NCMRWF / Ministry of Earth Sciences — prototype dashboard")


# ═══════════════════════════════════════════════════════════════════
# MODEL & DATA LOADING (cached — runs only once)
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model & data …")
def load_state():
    cfg = load_cfg()
    feat_dir = P(cfg, "features")
    meta = json.load(open(feat_dir / "feature_meta.json"))
    feature_cols = meta["feature_columns"]
    err_cols = [f"z_err_{v}" for v in meta["error_vars"]]
    leads = meta["leads"]
    hist_cols, lead_cols = split_feature_cols(feature_cols)

    g = np.load(feat_dir / "graph.npz")
    device = torch.device("cpu")                       # Streamlit Cloud = CPU only
    model = BustCastModel(
        n_hist_feat=len(hist_cols), n_lead_feat=len(lead_cols),
        n_points=len(g["point_id"]), n_leads=len(leads),
        n_error_vars=len(err_cols), cfg=cfg,
        coords_norm=norm_coords(g["region_centroid"]),
        region_pool=g["region_pool"],
    ).to(device)
    ckpt = Path(cfg["paths"]["outputs"]) / "checkpoints" / "stage3_best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    calib_path = Path(cfg["paths"]["outputs"]) / "calibration" / "calibration_report.json"
    calibration = json.load(open(calib_path)) if calib_path.exists() else {}

    df = pl.read_parquet(feat_dir / "samples.parquet")
    medians = {c: (df.filter(pl.col("split") == "train")[c].median() or 0.0) for c in feature_cols}

    return dict(
        cfg=cfg, meta=meta, feature_cols=feature_cols,
        hist_cols=hist_cols, lead_cols=lead_cols,
        err_cols=err_cols, leads=leads, g=g,
        device=device, model=model, calibration=calibration,
        df=df, medians=medians,
        pid_to_slot={int(p): i for i, p in enumerate(g["point_id"])},
        nbr=torch.from_numpy(g["nbr"]).long().to(device),
        edge_feat=torch.from_numpy(g["edge_feat"]).float().to(device),
    )


# ═══════════════════════════════════════════════════════════════════
# INFERENCE HELPERS (same logic as api.py, but no HTTP)
# ═══════════════════════════════════════════════════════════════════

def risk_category(p, cfg):
    edges, labels = cfg["risk"]["edges"], cfg["risk"]["labels"]
    for e, lab in zip(edges, labels):
        if p <= e:
            return lab
    return labels[-1]


def get_init_dates(state, split):
    df = state["df"]
    if split:
        df = df.filter(pl.col("split") == split)
    return sorted(str(d) for d in df["init_date"].unique().to_list())


@st.cache_data(show_spinner="Running forecast …")
def run_forecast(_state, init_date, explain_threshold):
    """Run model inference for one init_date. _state is unhashed (leading _)."""
    state = _state
    df = state["df"]
    day = df.filter(pl.col("init_date").cast(pl.Utf8) == init_date)
    if day.height == 0:
        return None

    g, leads = state["g"], state["leads"]
    n_points = len(g["point_id"])
    hist_cols, lead_cols = state["hist_cols"], state["lead_cols"]
    medians = state["medians"]

    Xh = np.zeros((n_points, len(hist_cols)), dtype=np.float32)
    Xl = np.zeros((n_points, len(leads), len(lead_cols)), dtype=np.float32)
    hist_done = np.zeros(n_points, dtype=bool)

    for row in day.iter_rows(named=True):
        slot = state["pid_to_slot"].get(int(row["point_id"]))
        if slot is None or row["lead_day"] not in leads:
            continue
        li = leads.index(row["lead_day"])
        if not hist_done[slot]:
            for fi, fc in enumerate(hist_cols):
                v = row.get(fc)
                Xh[slot, fi] = v if v is not None else medians.get(fc, 0.0)
            hist_done[slot] = True
        for fi, fc in enumerate(lead_cols):
            v = row.get(fc)
            Xl[slot, li, fi] = v if v is not None else medians.get(fc, 0.0)

    device = state["device"]
    Xh_t = torch.from_numpy(Xh).unsqueeze(0).to(device)
    Xl_t = torch.from_numpy(Xl).unsqueeze(0).to(device)
    with torch.no_grad():
        err, errvar, bust_logit = state["model"](Xh_t, Xl_t, state["nbr"], state["edge_feat"])
    err = err[0].cpu().numpy()
    prob = torch.sigmoid(bust_logit)[0].cpu().numpy()

    cfg = state["cfg"]
    grid_predictions, region_rows = [], []
    coords = g["coords"]
    region_label = g["region_label"]

    for slot in range(n_points):
        if not hist_done[slot]:
            continue
        pid = int(g["point_id"][slot])
        for li, lead in enumerate(leads):
            p = float(prob[slot, li])
            grid_predictions.append(dict(
                point_id=pid,
                latitude=float(coords[slot, 0]),
                longitude=float(coords[slot, 1]),
                lead_day=lead,
                predicted_error=float(err[slot, li]),
                bust_probability=p,
                confidence=1 - p,
                risk_category=risk_category(p, cfg),
            ))

    for r in np.unique(region_label):
        sel = np.where(region_label == r)[0]
        sel = [s for s in sel if hist_done[s]]
        if not sel:
            continue
        for li, lead in enumerate(leads):
            pe = err[sel, li]
            pp = prob[sel, li]
            region_rows.append(dict(
                region_id=int(r), lead_day=lead,
                mean_predicted_error=float(pe.mean()),
                max_predicted_error=float(pe.max()),
                mean_bust_probability=float(pp.mean()),
                max_bust_probability=float(pp.max()),
                pct_high_risk=float((pp > cfg["risk"]["edges"][2]).mean()),
                confidence=float(1 - pp.mean()),
            ))

    explanations = []
    try:
        from bustcast.explain import explain_for_request
        explanations = explain_for_request(state, Xh, Xl, hist_done, explain_threshold)
    except Exception:
        pass

    return dict(
        grid_predictions=grid_predictions,
        regions=region_rows,
        explanations=explanations,
    )


# ═══════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════

state = load_state()

col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    split = st.selectbox("Data split to browse", ["test", "calib", "val", "train"], index=0)

dates = get_init_dates(state, split)
if not dates:
    st.warning(f"No init dates available for split='{split}'.")
    st.stop()

with col2:
    init_date = st.selectbox("Forecast initialization date", dates, index=len(dates) - 1)
with col3:
    explain_threshold = st.slider("Explanation flag threshold (P(bust) >)", 0.1, 0.9, 0.5, 0.05)

data = run_forecast(state, init_date, explain_threshold)
if data is None:
    st.error(f"No data found for init_date={init_date}.")
    st.stop()

df = pd.DataFrame(data["grid_predictions"])

lead = st.select_slider("Lead day", options=sorted(df["lead_day"].unique().tolist()))
sub = df[df["lead_day"] == lead]

st.subheader(f"Forecast issued {init_date} — Day {lead}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Points shown", len(sub))
m2.metric("Mean bust probability", f"{sub['bust_probability'].mean():.2f}")
m3.metric("Mean confidence", f"{sub['confidence'].mean():.2f}")
m4.metric("High/very-high risk points", int((sub["risk_category"].isin(["high", "very_high"])).sum()))

tab_conf, tab_bust, tab_err, tab_areas, tab_explain = st.tabs(
    ["Forecast Confidence Map", "Bust Probability Map", "Predicted Error Map",
     "Error-Prone Areas", "Explanation Panel"])

with tab_conf:
    fig = px.scatter_geo(sub, lat="latitude", lon="longitude", color="confidence",
                          color_continuous_scale="Greens", scope="asia",
                          hover_data=["point_id", "risk_category"], title=f"Confidence — Day {lead}")
    fig.update_geos(lataxis_range=[5, 38], lonaxis_range=[65, 100])
    st.plotly_chart(fig, use_container_width=True)

with tab_bust:
    fig = px.scatter_geo(sub, lat="latitude", lon="longitude", color="bust_probability",
                          color_continuous_scale="Reds", scope="asia",
                          hover_data=["point_id", "risk_category"], title=f"P(bust) — Day {lead}")
    fig.update_geos(lataxis_range=[5, 38], lonaxis_range=[65, 100])
    st.plotly_chart(fig, use_container_width=True)

with tab_err:
    fig = px.scatter_geo(sub, lat="latitude", lon="longitude", color="predicted_error",
                          color_continuous_scale="Magma", scope="asia",
                          hover_data=["point_id"], title=f"Predicted composite error (z) — Day {lead}")
    fig.update_geos(lataxis_range=[5, 38], lonaxis_range=[65, 100])
    st.plotly_chart(fig, use_container_width=True)

with tab_areas:
    reg = pd.DataFrame(data["regions"])
    reg = reg[reg["lead_day"] == lead].sort_values("mean_bust_probability", ascending=False)
    st.markdown("**Region-wise aggregation** (KMeans-clustered regions over the 900 points)")
    st.dataframe(reg, use_container_width=True, hide_index=True)
    st.markdown("**High-risk points at this lead**")
    st.dataframe(sub[sub["risk_category"].isin(["high", "very_high"])]
                 .sort_values("bust_probability", ascending=False), use_container_width=True, hide_index=True)

with tab_explain:
    ex = pd.DataFrame(data["explanations"])
    ex = ex[ex["lead_day"] == lead] if len(ex) else ex
    if len(ex) == 0:
        st.info("No points crossed the explanation threshold at this lead day. Lower the "
                 "threshold in the controls above to see more.")
    else:
        for _, row in ex.iterrows():
            with st.expander(f"Point {row['point_id']} — P(bust)={row['bust_probability']:.2f}"):
                st.write("Dominant meteorological factors (input-gradient saliency):")
                for f in row["dominant_factors"]:
                    st.markdown(f"- {f}")

st.divider()
st.caption("Predicted error is the calibrated, z-normalised composite error (see config.yaml "
           "`targets`). Confidence = 1 - calibrated P(bust). This is a prototype for operational "
           "review, not an official NCMRWF product.")
