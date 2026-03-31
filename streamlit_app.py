"""
streamlit_app.py - UFC Fight Predictor Web UI

Cascading filter flow:
  1. Gender (Male / Female / All)
  2. Weight Class (filtered by gender)
  3. Fighter selection (filtered by weight class)

Supports two prediction modes:
  - Historical: exact matchup found in dataset (with-odds model)
  - Synthetic:  any two fighters (no-odds model)
"""

import json
import pathlib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.predict import predict_matchup

# ---- Page config ----
st.set_page_config(
    page_title="UFC Fight Predictor",
    page_icon="🥊",
    layout="centered",
)

# ---- Constants ----
CLEANED_PATH = pathlib.Path(__file__).parent / "data" / "processed" / "ufc_cleaned.csv"
DISPLAY_STATS_PATH = pathlib.Path(__file__).parent / "reports" / "feature_display_stats.json"

RED_COLOR = "#E63946"
BLUE_COLOR = "#457B9D"
BG_COLOR = "#0E1117"
TEXT_COLOR = "#FAFAFA"

MALE_WEIGHT_CLASSES = [
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
    "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
    "Catch Weight",
]
FEMALE_WEIGHT_CLASSES = [
    "Women's Strawweight", "Women's Flyweight",
    "Women's Bantamweight", "Women's Featherweight",
]


# ---- Cached data loaders ----
@st.cache_data
def load_fighter_index() -> pd.DataFrame:
    """
    Build a fighter index: each fighter's most recent weight class and gender.
    Returns DataFrame with columns: Fighter, WeightClass, Gender.
    """
    df = pd.read_csv(CLEANED_PATH, usecols=["RedFighter", "BlueFighter", "WeightClass", "Gender", "Date"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    red = df[["RedFighter", "WeightClass", "Gender", "Date"]].rename(columns={"RedFighter": "Fighter"})
    blue = df[["BlueFighter", "WeightClass", "Gender", "Date"]].rename(columns={"BlueFighter": "Fighter"})
    combined = pd.concat([red, blue], ignore_index=True)

    # Keep the most recent appearance per fighter
    combined = combined.sort_values("Date", ascending=False).drop_duplicates(subset="Fighter", keep="first")
    return combined[["Fighter", "WeightClass", "Gender"]].reset_index(drop=True)


fighter_index = load_fighter_index()


@st.cache_data
def load_display_stats() -> dict:
    """Load feature display normalization stats (mean/std per feature)."""
    if DISPLAY_STATS_PATH.exists():
        with open(DISPLAY_STATS_PATH) as f:
            return json.load(f)
    return {}


display_stats = load_display_stats()

# Reverse mapping: display label -> column name (from predict.py EXPLAINABILITY_DIFFS)
DIFF_LABEL_TO_COL = {
    "Reach": "ReachDif",
    "Age": "AgeDif",
    "Sig. Strikes": "SigStrDif",
    "Takedowns": "AvgTDDif",
    "Sub. Attempts": "AvgSubAttDif",
    "Height": "HeightDif",
    "Win Streak": "WinStreakDif",
    "WC Rank": "wc_rank_diff",
    "P4P Rank": "pfp_rank_diff",
}


def get_filtered_fighters(gender: str, weight_class: str) -> list[str]:
    """Filter fighters by gender and weight class selection."""
    filtered = fighter_index.copy()

    if gender != "All":
        filtered = filtered[filtered["Gender"] == gender]

    if weight_class != "All Divisions":
        filtered = filtered[filtered["WeightClass"] == weight_class]

    return sorted(filtered["Fighter"].unique().tolist())


def get_weight_classes_for_gender(gender: str) -> list[str]:
    """Return weight classes available for the selected gender."""
    if gender == "MALE":
        return MALE_WEIGHT_CLASSES
    elif gender == "FEMALE":
        return FEMALE_WEIGHT_CLASSES
    else:
        return MALE_WEIGHT_CLASSES + FEMALE_WEIGHT_CLASSES


# ---- Header ----
st.title("🥊 UFC Fight Predictor")
st.markdown("Predict the outcome of a UFC fight using machine learning.")

# ---- Step 1: Gender + Weight Class row ----
filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    gender = st.selectbox("Division", options=["All", "MALE", "FEMALE"], index=0,
                          format_func=lambda x: {"All": "All Fighters", "MALE": "Men's", "FEMALE": "Women's"}[x])

with filter_col2:
    wc_options = ["All Divisions"] + get_weight_classes_for_gender(gender)
    weight_class = st.selectbox("Weight Class", options=wc_options, index=0)

# ---- Step 2: Filtered fighter lists ----
available_fighters = get_filtered_fighters(gender, weight_class)

if len(available_fighters) == 0:
    st.warning("No fighters found for this filter combination.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    red_fighter = st.selectbox("Red Corner", options=available_fighters, index=None,
                               placeholder="Choose a fighter...")

with col2:
    blue_fighter = st.selectbox("Blue Corner", options=available_fighters, index=None,
                                placeholder="Choose a fighter...")

# ---- Step 3: Fight context ----
with st.expander("Fight Context (optional)"):
    ctx_col1, ctx_col2 = st.columns(2)

    with ctx_col1:
        title_bout = st.checkbox("Title Bout", value=False)

    with ctx_col2:
        num_rounds = st.selectbox("Rounds", options=[3, 5], index=0)

# ---- Cross-division guardrail (rendered before button so state persists) ----
cross_division = False
cross_div_override = False

if red_fighter and blue_fighter and red_fighter != blue_fighter:
    red_wc_row = fighter_index[fighter_index["Fighter"] == red_fighter]
    blue_wc_row = fighter_index[fighter_index["Fighter"] == blue_fighter]
    red_wc = red_wc_row["WeightClass"].iloc[0] if not red_wc_row.empty else None
    blue_wc = blue_wc_row["WeightClass"].iloc[0] if not blue_wc_row.empty else None

    if red_wc and blue_wc and red_wc != blue_wc:
        cross_division = True
        st.warning(
            f"⚠️ **Cross-division matchup detected**: "
            f"{red_fighter} ({red_wc}) vs {blue_fighter} ({blue_wc}). "
            f"The model was not trained on cross-weight-class fights — "
            f"predictions may be unrealistic."
        )
        cross_div_override = st.checkbox(
            "Allow cross-division prediction (for fun)", value=False
        )

# ---- Predict button ----
if st.button("Predict", type="primary", use_container_width=True):
    if not red_fighter or not blue_fighter:
        st.warning("Please select both fighters.")
    elif red_fighter == blue_fighter:
        st.warning("Please select two different fighters.")
    elif cross_division and not cross_div_override:
        st.info("Enable the cross-division checkbox above to run this prediction.")
    else:
        # Use selected weight class if specific, otherwise auto-detect
        wc_override = None if weight_class == "All Divisions" else weight_class

        with st.spinner("Running model inference..."):
            try:
                result = predict_matchup(
                    red_fighter, blue_fighter,
                    weight_class=wc_override,
                    title_bout=title_bout,
                    num_rounds=num_rounds,
                )
            except (SystemExit, FileNotFoundError, ValueError) as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        # Show persistent warning if cross-division override was used on synthetic
        if cross_division and "synthetic" in result.get("mode", ""):
            st.warning(
                "🚧 **Cross-division matchup**: model not trained on these scenarios; "
                "results may be unrealistic."
            )

        # Show OOD clipping warnings (synthetic mode only)
        for warn_msg in result.get("ood_warnings", []):
            st.warning(warn_msg)

        prob_red = result["proba_red"]
        prob_blue = 1.0 - prob_red
        winner = result["winner_name"]
        corner = result["predicted_winner"]
        mode = result["mode"]
        model_variant = result.get("model_variant", "unknown")
        diffs = result.get("diffs", {})

        # ---- Results ----
        st.markdown("---")
        st.subheader("Prediction")

        if "historical" in mode:
            st.caption(
                f"📋 **Mode: Historical** (with odds) · "
                f"Model: `{model_variant}` · "
                f"Fight date: {result.get('fight_date', '?')}"
            )
        else:
            st.caption(
                f"🔧 **Mode: Synthetic** (no odds) · "
                f"Model: `{model_variant}` · "
                f"Red stats: {result.get('red_data_from', '?')} · "
                f"Blue stats: {result.get('blue_data_from', '?')}"
            )

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

        ax1.barh(0, prob_red, color=RED_COLOR, height=0.6, label=red_fighter)
        ax1.barh(0, prob_blue, left=prob_red, color=BLUE_COLOR, height=0.6, label=blue_fighter)

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
            nonzero_diffs = {k: v for k, v in diffs.items() if abs(v) > 0.01}

            if nonzero_diffs:
                # Pick stats set based on mode
                if "historical" in mode:
                    feat_stats = display_stats.get("v1_with_odds", {}).get("stats", {})
                else:
                    feat_stats = display_stats.get("v2_no_odds", {}).get("stats", {})

                # Compute z-scores (labels are display names, stats keys are column names)
                z_diffs = {}
                for k, v in nonzero_diffs.items():
                    col_name = DIFF_LABEL_TO_COL.get(k, k)  # map display label -> column name
                    fs = feat_stats.get(col_name)
                    if fs and fs.get("std", 0) > 0:
                        z = (v - fs["mean"]) / fs["std"]
                        z_diffs[k] = max(-3.0, min(3.0, z))

                if z_diffs:
                    st.markdown("#### Matchup Advantages (standardized)")
                    st.caption("Values are z-scores vs training distribution. Positive (red) favors Red, negative (blue) favors Blue.")

                    labels = list(z_diffs.keys())
                    values = list(z_diffs.values())
                    colors = [RED_COLOR if v > 0 else BLUE_COLOR for v in values]

                    fig2, ax2 = plt.subplots(figsize=(8, max(2.0, len(labels) * 0.45)))
                    fig2.patch.set_facecolor(BG_COLOR)
                    ax2.set_facecolor(BG_COLOR)

                    bars = ax2.barh(labels, values, color=colors, height=0.6, edgecolor="none")

                    for bar, val in zip(bars, values):
                        x_pos = bar.get_width()
                        ha = "left" if val >= 0 else "right"
                        offset = 0.1 if val >= 0 else -0.1
                        ax2.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                                 f"{val:+.2f}σ", ha=ha, va="center",
                                 color=TEXT_COLOR, fontsize=10)

                    ax2.set_xlim(-3.5, 3.5)
                    ax2.axvline(0, color="#555555", linewidth=0.8, linestyle="-")
                    ax2.tick_params(axis="y", colors=TEXT_COLOR, labelsize=11)
                    ax2.tick_params(axis="x", colors="#777777", labelsize=9)
                    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

                    for spine in ax2.spines.values():
                        spine.set_visible(False)

                    fig2.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)

                # Optional raw diffs toggle
                if st.checkbox("Show raw diff values", value=False):
                    labels_raw = list(nonzero_diffs.keys())
                    values_raw = list(nonzero_diffs.values())
                    colors_raw = [RED_COLOR if v > 0 else BLUE_COLOR for v in values_raw]

                    fig3, ax3 = plt.subplots(figsize=(8, max(2.0, len(labels_raw) * 0.45)))
                    fig3.patch.set_facecolor(BG_COLOR)
                    ax3.set_facecolor(BG_COLOR)

                    bars3 = ax3.barh(labels_raw, values_raw, color=colors_raw, height=0.6, edgecolor="none")

                    for bar, val in zip(bars3, values_raw):
                        x_pos = bar.get_width()
                        ha = "left" if val >= 0 else "right"
                        offset = 0.3 if val >= 0 else -0.3
                        ax3.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                                 f"{val:+.1f}", ha=ha, va="center",
                                 color=TEXT_COLOR, fontsize=10)

                    ax3.axvline(0, color="#555555", linewidth=0.8, linestyle="-")
                    ax3.tick_params(axis="y", colors=TEXT_COLOR, labelsize=11)
                    ax3.tick_params(axis="x", colors="#777777", labelsize=9)
                    ax3.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

                    for spine in ax3.spines.values():
                        spine.set_visible(False)

                    fig3.tight_layout()
                    st.pyplot(fig3, use_container_width=True)
                    plt.close(fig3)

        # ---- Footer caption ----
        if "historical" in mode:
            st.caption(f"Based on fight data from {result['fight_date']}")
        else:
            st.caption(
                f"Red stats from {result.get('red_data_from', '?')} | "
                f"Blue stats from {result.get('blue_data_from', '?')}"
            )
