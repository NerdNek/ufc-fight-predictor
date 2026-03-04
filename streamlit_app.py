"""
streamlit_app.py - UFC Fight Predictor Web UI

Loads the dataset to populate fighter dropdowns,
runs model inference via predict_matchup(), and displays results
with win probability and matchup advantage charts.

Supports two prediction modes:
  - Historical: exact matchup found in dataset (with-odds model)
  - Synthetic:  any two fighters (no-odds model)
"""

import pathlib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.predict import predict_matchup, get_all_fighter_names

# ---- Page config ----
st.set_page_config(
    page_title="UFC Fight Predictor",
    page_icon="🥊",
    layout="centered",
)

# ---- Cached data loader ----
CLEANED_PATH = pathlib.Path(__file__).parent / "data" / "processed" / "ufc_cleaned.csv"

# Chart colors
RED_COLOR = "#E63946"
BLUE_COLOR = "#457B9D"
BG_COLOR = "#0E1117"
TEXT_COLOR = "#FAFAFA"


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
        diffs = result.get("diffs", {})

        # ---- Results display ----
        st.markdown("---")
        st.subheader("Prediction")

        # Mode badge
        if mode == "historical":
            st.caption("**Historical mode** -- exact matchup found in dataset (with-odds model)")
        else:
            st.caption("**Synthetic mode** -- matchup constructed from individual fighter stats (no-odds model)")

        m1, m2 = st.columns(2)
        m1.metric("P(Red wins)", f"{prob_red:.1%}")
        m2.metric("P(Blue wins)", f"{prob_blue:.1%}")

        if corner == "Red":
            st.success(f"**{winner}** (Red corner) is the predicted winner")
        else:
            st.info(f"**{winner}** (Blue corner) is the predicted winner")

        # ---- Graph 1: Win Probability Bar ----
        st.markdown("#### Win Probability")

        fig1, ax1 = plt.subplots(figsize=(8, 1.2))
        fig1.patch.set_facecolor(BG_COLOR)
        ax1.set_facecolor(BG_COLOR)

        # Stacked horizontal bar
        ax1.barh(0, prob_red, color=RED_COLOR, height=0.6, label=red_fighter)
        ax1.barh(0, prob_blue, left=prob_red, color=BLUE_COLOR, height=0.6, label=blue_fighter)

        # Labels inside bars
        if prob_red >= 0.12:
            ax1.text(prob_red / 2, 0, f"{prob_red:.0%}",
                     ha="center", va="center", color="white", fontsize=13, fontweight="bold")
        if prob_blue >= 0.12:
            ax1.text(prob_red + prob_blue / 2, 0, f"{prob_blue:.0%}",
                     ha="center", va="center", color="white", fontsize=13, fontweight="bold")

        ax1.set_xlim(0, 1)
        ax1.set_yticks([])
        ax1.set_xticks([])
        for spine in ax1.spines.values():
            spine.set_visible(False)

        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2,
                   frameon=False, fontsize=10, labelcolor=TEXT_COLOR)

        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        # ---- Graph 2: Matchup Advantage Diffs ----
        if diffs:
            # Filter out zero diffs for cleaner chart
            nonzero_diffs = {k: v for k, v in diffs.items() if abs(v) > 0.01}

            if nonzero_diffs:
                st.markdown("#### Matchup Advantages")
                st.caption("Positive (red) = Red corner advantage. Negative (blue) = Blue corner advantage.")

                labels = list(nonzero_diffs.keys())
                values = list(nonzero_diffs.values())
                colors = [RED_COLOR if v > 0 else BLUE_COLOR for v in values]

                fig2, ax2 = plt.subplots(figsize=(8, max(2.0, len(labels) * 0.45)))
                fig2.patch.set_facecolor(BG_COLOR)
                ax2.set_facecolor(BG_COLOR)

                bars = ax2.barh(labels, values, color=colors, height=0.6, edgecolor="none")

                # Value labels at bar ends
                for bar, val in zip(bars, values):
                    x_pos = bar.get_width()
                    ha = "left" if val >= 0 else "right"
                    offset = 0.3 if val >= 0 else -0.3
                    ax2.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                             f"{val:+.1f}", ha=ha, va="center",
                             color=TEXT_COLOR, fontsize=10)

                ax2.axvline(0, color="#555555", linewidth=0.8, linestyle="-")
                ax2.tick_params(axis="y", colors=TEXT_COLOR, labelsize=11)
                ax2.tick_params(axis="x", colors="#777777", labelsize=9)
                ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

                for spine in ax2.spines.values():
                    spine.set_visible(False)

                fig2.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

        # ---- Footer caption ----
        if mode == "historical":
            st.caption(f"Based on fight data from {result['fight_date']}")
        else:
            st.caption(
                f"Red stats from {result.get('red_data_from', '?')} | "
                f"Blue stats from {result.get('blue_data_from', '?')}"
            )
