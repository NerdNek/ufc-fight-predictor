"""
streamlit_app.py - UFC Fight Predictor Web UI

Loads the dataset to populate fighter dropdowns,
runs model inference via predict_matchup(), and displays results.

Supports two prediction modes:
  - Historical: exact matchup found in dataset (with-odds model)
  - Synthetic:  any two fighters (no-odds model)
"""

import pathlib
import streamlit as st
import pandas as pd

from src.predict import predict_matchup, get_all_fighter_names

# ---- Page config ----
st.set_page_config(
    page_title="UFC Fight Predictor",
    page_icon="🥊",
    layout="centered",
)

# ---- Cached data loader ----
CLEANED_PATH = pathlib.Path(__file__).parent / "data" / "processed" / "ufc_cleaned.csv"


@st.cache_data
def load_fighter_names() -> list[str]:
    """Return a sorted list of unique fighter names from the dataset."""
    df = pd.read_csv(CLEANED_PATH, usecols=["RedFighter", "BlueFighter"])
    fighters = sorted(
        set(df["RedFighter"].dropna().unique()) | set(df["BlueFighter"].dropna().unique())
    )
    return fighters


@st.cache_data
def load_weight_classes() -> list[str]:
    """Return sorted list of unique weight classes from the dataset."""
    df = pd.read_csv(CLEANED_PATH, usecols=["WeightClass"])
    return sorted(df["WeightClass"].dropna().unique().tolist())


fighters = load_fighter_names()
weight_classes = load_weight_classes()

# ---- Header ----
st.title("🥊 UFC Fight Predictor")
st.markdown("Predict the outcome of a UFC fight using machine learning.")

# ---- Fighter Inputs ----
col1, col2 = st.columns(2)

with col1:
    red_fighter = st.selectbox("Red Corner", options=fighters, index=None, placeholder="Choose a fighter…")

with col2:
    blue_fighter = st.selectbox("Blue Corner", options=fighters, index=None, placeholder="Choose a fighter…")

# ---- Fight Context (collapsible) ----
with st.expander("⚙️ Fight Context (optional)"):
    ctx_col1, ctx_col2, ctx_col3 = st.columns(3)

    with ctx_col1:
        weight_class = st.selectbox(
            "Weight Class",
            options=["Auto-detect"] + weight_classes,
            index=0,
        )

    with ctx_col2:
        title_bout = st.checkbox("Title Bout", value=False)

    with ctx_col3:
        num_rounds = st.selectbox("Rounds", options=[3, 5], index=0)

# ---- Predict button ----
if st.button("Predict", type="primary", use_container_width=True):
    if not red_fighter or not blue_fighter:
        st.warning("Please select both fighters.")
    elif red_fighter == blue_fighter:
        st.warning("Please select two different fighters.")
    else:
        wc = None if weight_class == "Auto-detect" else weight_class

        with st.spinner("Running model inference…"):
            try:
                result = predict_matchup(
                    red_fighter, blue_fighter,
                    weight_class=wc,
                    title_bout=title_bout,
                    num_rounds=num_rounds,
                )
            except (SystemExit, FileNotFoundError, ValueError) as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        prob_red = result["proba_red"]
        prob_blue = 1.0 - prob_red
        winner = result["winner_name"]
        corner = result["predicted_winner"]
        mode = result["mode"]

        # ---- Results display ----
        st.markdown("---")
        st.subheader("Prediction")

        # Mode badge
        if mode == "historical":
            st.caption(f"📊 **Historical mode** — exact matchup found in dataset (with-odds model)")
        else:
            st.caption(f"🔮 **Synthetic mode** — matchup constructed from individual fighter stats (no-odds model)")

        m1, m2 = st.columns(2)
        m1.metric("P(Red wins)", f"{prob_red:.1%}")
        m2.metric("P(Blue wins)", f"{prob_blue:.1%}")

        if corner == "Red":
            st.success(f"🏆 **{winner}** (Red corner) is the predicted winner")
        else:
            st.info(f"🏆 **{winner}** (Blue corner) is the predicted winner")

        if mode == "historical":
            st.caption(f"Based on fight data from {result['fight_date']}")
        else:
            st.caption(
                f"Red stats from {result.get('red_data_from', '?')} · "
                f"Blue stats from {result.get('blue_data_from', '?')}"
            )
