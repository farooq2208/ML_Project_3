"""
California House Price Predictor — Streamlit Web Application
=============================================================
Predicts California median house values using an XGBoost regression
pipeline trained on the California Housing dataset.

Feature engineering is performed inside this app (same transforms as
the training notebook) before the pre-trained pipeline is called.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = "California_House_Price_Prediction.pkl"

OCEAN_PROXIMITY_OPTIONS = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

# California geographic bounds (from training data)
LON_MIN, LON_MAX = -124.35, -114.31
LAT_MIN, LAT_MAX =   32.54,   41.95

# Price bands for result colour coding
PRICE_BANDS = [
    (150_000,  "#388e3c", "#e8f5e9", "#a5d6a7", "Budget-friendly"),
    (300_000,  "#1565c0", "#e3f2fd", "#90caf9", "Mid-range"),
    (450_000,  "#6a1b9a", "#f3e5f5", "#ce93d8", "Premium"),
    (float("inf"), "#b71c1c", "#fce4ec", "#f48fb1", "Luxury"),
]


# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """Load the pre-trained XGBoost pipeline from disk."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(
            f"❌ Model file `{MODEL_PATH}` not found. "
            "Make sure it lives in the same directory as `app.py`."
        )
        st.stop()


# ── Helper: feature engineering (mirrors the training notebook) ───────────────
def engineer_features(
    longitude: float,
    latitude: float,
    housing_median_age: float,
    total_rooms: float,
    total_bedrooms: float,
    population: float,
    households: float,
    median_income: float,
    ocean_proximity: str,
) -> pd.DataFrame:
    """
    Reproduce the exact feature engineering steps from the training notebook:
      1. Derive three ratio features.
      2. Drop the four raw count columns.
      3. Return a single-row DataFrame with plain object dtypes for strings
         (prevents ArrowStringDtype errors on Streamlit Cloud).
    """
    rooms_per_household      = total_rooms    / households
    bedrooms_per_room        = total_bedrooms / total_rooms
    population_per_household = population     / households

    return pd.DataFrame({
        # Columns must match the order seen during training
        "longitude":                pd.array([float(longitude)],                    dtype="float64"),
        "latitude":                 pd.array([float(latitude)],                     dtype="float64"),
        "housing_median_age":       pd.array([float(housing_median_age)],           dtype="float64"),
        "median_income":            pd.array([float(median_income)],                dtype="float64"),
        "ocean_proximity":          pd.array([str(ocean_proximity)],                dtype="object"),
        "rooms_per_household":      pd.array([float(rooms_per_household)],          dtype="float64"),
        "bedrooms_per_room":        pd.array([float(bedrooms_per_room)],            dtype="float64"),
        "population_per_household": pd.array([float(population_per_household)],     dtype="float64"),
    })


# ── Helper: pick colour band for a given price ────────────────────────────────
def get_price_band(price: float) -> dict:
    for threshold, color, bg, border, label in PRICE_BANDS:
        if price < threshold:
            return {"color": color, "bg": bg, "border": border, "label": label}
    return {"color": "#b71c1c", "bg": "#fce4ec", "border": "#f48fb1", "label": "Luxury"}


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/cottage.png", width=80)
        st.title("California House Price Predictor")
        st.markdown("---")

        st.markdown(
            """
            ### About
            This app uses an **XGBoost** regression model trained on
            California census housing data to estimate the **median house
            value** for a given block group.

            ### How to use
            1. Enter location coordinates or drag the map pin.
            2. Fill in the housing and population details.
            3. Select the proximity to the ocean.
            4. Click **Predict House Price**.

            ### Price Bands
            | Band | Range |
            |------|-------|
            | 🟢 Budget-friendly | < $150 000 |
            | 🔵 Mid-range | $150 000 – $300 000 |
            | 🟣 Premium | $300 000 – $450 000 |
            | 🔴 Luxury | > $450 000 |

            ### Disclaimer
            > Predictions are **estimates only** and should not be used
            > for financial or real-estate decisions without consulting
            > a qualified professional.
            """
        )

        st.markdown("---")
        st.markdown(
            """
            ### Model Details
            - **Algorithm:** XGBoost Regressor
            - **Tuning:** 5-fold GridSearchCV
            - **Metric:** R² Score
            - **Test R²:** ~0.83
            - **Test MAE:** ~$26 500
            - **Dataset:** 20 640 California block groups
            """
        )

        st.markdown("---")
        st.caption("Built with [Streamlit](https://streamlit.io) 🎈")


# ── Input validation ──────────────────────────────────────────────────────────
def validate_inputs(
    longitude, latitude, total_rooms, total_bedrooms, population, households
) -> list[str]:
    """Return a list of human-readable error strings (empty = all valid)."""
    errors = []

    if not (LON_MIN <= longitude <= LON_MAX):
        errors.append(
            f"Longitude must be between {LON_MIN} and {LON_MAX} "
            "(California range)."
        )
    if not (LAT_MIN <= latitude <= LAT_MAX):
        errors.append(
            f"Latitude must be between {LAT_MIN} and {LAT_MAX} "
            "(California range)."
        )
    if total_bedrooms > total_rooms:
        errors.append("Total bedrooms cannot exceed total rooms.")
    if households > population:
        errors.append("Households cannot exceed population.")
    if households < 1:
        errors.append("Households must be at least 1.")
    if total_rooms < 1:
        errors.append("Total rooms must be at least 1.")

    return errors


# ── Input form ────────────────────────────────────────────────────────────────
def render_input_form() -> dict:
    """Render the user-input form and return all raw values."""

    st.header("📋 Enter Housing Block Details")
    st.markdown(
        "All values refer to a **census block group** — a small geographic "
        "area containing roughly 600–3 000 people."
    )

    # ── Row 1: Location ───────────────────────────────────────────────────────
    st.subheader("📍 Location")
    loc_col1, loc_col2, loc_col3 = st.columns(3)

    with loc_col1:
        longitude = st.number_input(
            "Longitude",
            min_value=float(LON_MIN),
            max_value=float(LON_MAX),
            value=-118.49,
            step=0.01,
            format="%.4f",
            help=(
                f"West–east coordinate. California range: {LON_MIN} to {LON_MAX}. "
                "Los Angeles ≈ -118.49, San Francisco ≈ -122.42."
            ),
        )

    with loc_col2:
        latitude = st.number_input(
            "Latitude",
            min_value=float(LAT_MIN),
            max_value=float(LAT_MAX),
            value=34.02,
            step=0.01,
            format="%.4f",
            help=(
                f"North–south coordinate. California range: {LAT_MIN} to {LAT_MAX}. "
                "Los Angeles ≈ 34.02, San Francisco ≈ 37.77."
            ),
        )

    with loc_col3:
        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            options=OCEAN_PROXIMITY_OPTIONS,
            index=0,
            help=(
                "**<1H OCEAN** — within 1 hour's drive of the coast | "
                "**NEAR OCEAN** — directly on the coast | "
                "**NEAR BAY** — near San Francisco Bay | "
                "**INLAND** — interior California | "
                "**ISLAND** — island location (rare)"
            ),
        )

    # ── Row 2: Housing characteristics ───────────────────────────────────────
    st.subheader("🏘️ Housing Characteristics")
    house_col1, house_col2, house_col3 = st.columns(3)

    with house_col1:
        housing_median_age = st.slider(
            "Median House Age (years)",
            min_value=1,
            max_value=52,
            value=28,
            step=1,
            help="Median age of houses in the block group. Capped at 52 in this dataset.",
        )

    with house_col2:
        total_rooms = st.number_input(
            "Total Rooms",
            min_value=1,
            max_value=40000,
            value=2127,
            step=50,
            help=(
                "Total number of rooms across all households in the block group. "
                "Typical range: 1 448 – 3 148. Median: 2 127."
            ),
        )

    with house_col3:
        total_bedrooms = st.number_input(
            "Total Bedrooms",
            min_value=1,
            max_value=7000,
            value=435,
            step=10,
            help=(
                "Total number of bedrooms across all households. "
                "Must not exceed Total Rooms. Typical median: 435."
            ),
        )

    # ── Row 3: Population & income ────────────────────────────────────────────
    st.subheader("👥 Population & Income")
    pop_col1, pop_col2, pop_col3 = st.columns(3)

    with pop_col1:
        population = st.number_input(
            "Population",
            min_value=1,
            max_value=40000,
            value=1166,
            step=50,
            help="Total population in the block group. Typical range: 787 – 1 725.",
        )

    with pop_col2:
        households = st.number_input(
            "Households",
            min_value=1,
            max_value=7000,
            value=409,
            step=10,
            help=(
                "Number of households (occupied housing units) in the block group. "
                "Must not exceed Population. Typical median: 409."
            ),
        )

    with pop_col3:
        median_income = st.slider(
            "Median Income (tens of thousands USD)",
            min_value=0.5,
            max_value=15.0,
            value=3.5,
            step=0.1,
            format="%.1f",
            help=(
                "Median household income scaled to tens of thousands of dollars. "
                "E.g. 3.5 ≈ $35 000/year. Typical range: 2.5 – 4.7."
            ),
        )

    # ── Derived features preview ──────────────────────────────────────────────
    with st.expander("🔧 Preview derived features (auto-calculated)", expanded=False):
        st.caption(
            "These three features are automatically computed from your inputs "
            "before being passed to the model — exactly as done during training."
        )
        d_col1, d_col2, d_col3 = st.columns(3)
        safe_households = max(households, 1)
        safe_rooms      = max(total_rooms, 1)

        d_col1.metric(
            "Rooms per Household",
            f"{total_rooms / safe_households:.2f}",
            help="total_rooms / households",
        )
        d_col2.metric(
            "Bedrooms per Room",
            f"{total_bedrooms / safe_rooms:.3f}",
            help="total_bedrooms / total_rooms",
        )
        d_col3.metric(
            "Population per Household",
            f"{population / safe_households:.2f}",
            help="population / households",
        )

    return {
        "longitude":          longitude,
        "latitude":           latitude,
        "housing_median_age": housing_median_age,
        "total_rooms":        total_rooms,
        "total_bedrooms":     total_bedrooms,
        "population":         population,
        "households":         households,
        "median_income":      median_income,
        "ocean_proximity":    ocean_proximity,
    }


# ── Result display ────────────────────────────────────────────────────────────
def render_result(predicted_price: float, inputs: dict):
    """Render the prediction card, comparable-price reference, and input summary."""

    band = get_price_band(predicted_price)

    # ── Main prediction card ──────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: {band['bg']};
            border: 2px solid {band['border']};
            border-radius: 14px;
            padding: 28px 32px;
            margin-top: 12px;
        ">
            <p style="color:{band['color']}; font-weight:700; font-size:1rem;
                      margin:0 0 4px 0; text-transform:uppercase; letter-spacing:1px;">
                {band['label']} Property
            </p>
            <h1 style="color:{band['color']}; margin:0 0 6px 0; font-size:3rem;">
                ${predicted_price:,.0f}
            </h1>
            <p style="color:#555; margin:0; font-size:1rem;">
                Estimated median house value for this California block group.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    res_col1, res_col2 = st.columns([1, 1], gap="large")

    # ── Key metrics ───────────────────────────────────────────────────────────
    with res_col1:
        st.subheader("📊 Prediction Breakdown")

        st.metric("Predicted Median Value",    f"${predicted_price:,.0f}")
        st.metric("Model Typical Error (MAE)", "± $26,500",
                  help="Average absolute error on the held-out test set.")
        st.metric("Model R² (test set)",       "0.83",
                  help="83 % of price variance explained by the model.")

        # Confidence interval approximation (±1 MAE)
        lower = max(0, predicted_price - 26_500)
        upper = predicted_price + 26_500
        st.info(
            f"📐 **Approximate range:** ${lower:,.0f} – ${upper:,.0f}  \n"
            f"*(±1 × MAE around the point estimate)*"
        )

    # ── Context & interpretation ──────────────────────────────────────────────
    with res_col2:
        st.subheader("📍 Location Context")

        # Derived ratios
        safe_hh    = max(inputs["households"], 1)
        safe_rooms = max(inputs["total_rooms"], 1)
        rph  = inputs["total_rooms"]    / safe_hh
        bpr  = inputs["total_bedrooms"] / safe_rooms
        pph  = inputs["population"]     / safe_hh
        inc_k = inputs["median_income"] * 10  # tens of thousands → thousands

        st.markdown(
            f"""
            | Feature | Your Block | CA Median |
            |---------|-----------|-----------|
            | Rooms per household | {rph:.1f} | 5.2 |
            | Bedrooms per room | {bpr:.2f} | 0.20 |
            | People per household | {pph:.1f} | 2.8 |
            | Median income | ${inc_k:,.0f}K | $34.5K |
            | House age | {inputs['housing_median_age']} yrs | 29 yrs |
            | Ocean proximity | {inputs['ocean_proximity']} | — |
            """
        )

    # ── Input summary ─────────────────────────────────────────────────────────
    with st.expander("🔍 Review your submitted inputs"):
        display = {
            "Longitude":           inputs["longitude"],
            "Latitude":            inputs["latitude"],
            "Ocean Proximity":     inputs["ocean_proximity"],
            "Median House Age":    f"{inputs['housing_median_age']} yrs",
            "Total Rooms":         f"{inputs['total_rooms']:,}",
            "Total Bedrooms":      f"{inputs['total_bedrooms']:,}",
            "Population":          f"{inputs['population']:,}",
            "Households":          f"{inputs['households']:,}",
            "Median Income":       f"{inputs['median_income']:.1f} (×$10K)",
        }
        summary_df = pd.DataFrame(
            [{"Input Feature": k, "Value": str(v)} for k, v in display.items()]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ── Main application ──────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # ── Hero banner ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0d47a1 0%, #1b5e20 100%);
            border-radius: 14px;
            padding: 32px 36px;
            margin-bottom: 28px;
        ">
            <h1 style="color:white; margin:0 0 8px 0; font-size:2.3rem;">
                🏠 California House Price Predictor
            </h1>
            <p style="color:#c8e6c9; font-size:1.05rem; margin:0;">
                Enter details about a California census block group and our
                <strong>XGBoost</strong> model will estimate the median house
                value — trained on 20 000+ real block groups with an R² of 0.83.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model  = load_model()
    inputs = render_input_form()

    st.markdown("---")

    # ── Predict button ────────────────────────────────────────────────────────
    btn_col, _ = st.columns([1, 3])
    with btn_col:
        predict_clicked = st.button(
            "🔍 Predict House Price",
            type="primary",
            use_container_width=True,
        )

    if predict_clicked:
        errors = validate_inputs(
            inputs["longitude"],
            inputs["latitude"],
            inputs["total_rooms"],
            inputs["total_bedrooms"],
            inputs["population"],
            inputs["households"],
        )

        if errors:
            for err in errors:
                st.error(f"⚠️ {err}")
        else:
            # Build model-ready DataFrame (feature engineering applied here)
            input_df = engineer_features(
                longitude          = inputs["longitude"],
                latitude           = inputs["latitude"],
                housing_median_age = inputs["housing_median_age"],
                total_rooms        = inputs["total_rooms"],
                total_bedrooms     = inputs["total_bedrooms"],
                population         = inputs["population"],
                households         = inputs["households"],
                median_income      = inputs["median_income"],
                ocean_proximity    = inputs["ocean_proximity"],
            )

            with st.spinner("Running prediction…"):
                predicted_price = float(model.predict(input_df)[0])

            render_result(predicted_price, inputs)


if __name__ == "__main__":
    main()
