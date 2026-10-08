from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="AnimalSense AI", layout="wide")
st.title("AnimalSense AI")
st.caption("Contactless individual behavioural monitoring and early-warning prototype")
video = ROOT / "outputs/videos/animalsense_output.mp4"
summary = ROOT / "outputs/logs/animal_summary.csv"
alerts = ROOT / "outputs/logs/alerts.json"
log = ROOT / "outputs/logs/behaviour_log.csv"
if video.exists(): st.video(str(video))
else: st.info("Run: python main.py --source input/cattle_video.mp4")
if summary.exists():
    df = pd.read_csv(summary)
    c1,c2,c3 = st.columns(3)
    c1.metric("Tracked animals", len(df))
    c2.metric("Baselines ready", int(df["baseline_ready"].astype(str).str.lower().eq("true").sum()))
    c3.metric("Warnings", int(df["risk_level"].astype(str).str.contains("WARNING").sum()))
    st.subheader("Animal summary"); st.dataframe(df, use_container_width=True)
if alerts.exists():
    data = json.loads(alerts.read_text(encoding="utf-8"))
    st.subheader("Alerts")
    st.json(data) if data else st.success("No persistent alerts recorded.")
if log.exists():
    df = pd.read_csv(log)
    if not df.empty:
        st.subheader("Behaviour timeline")
        aid = st.selectbox("Animal", sorted(df["animal_id"].dropna().unique().tolist()))
        sub = df[df["animal_id"] == aid]
        st.line_chart(sub.set_index("timestamp")["deviation_score"])
        st.dataframe(sub.tail(100), use_container_width=True)
