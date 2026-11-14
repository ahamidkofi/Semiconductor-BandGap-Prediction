import pandas as pd
import streamlit as st
from predict import load_bundle, predict_gap

APP_TITLE = "Machine Learning Band Gap Predictor for Semiconductors & Inorganic Materials"
AUTHOR = "Hamid Kofi Abdul"

st.set_page_config(page_title=APP_TITLE, page_icon="🔬", layout="centered")


@st.cache_resource
def get_bundle():
    # Loaded once per session
    return load_bundle()


# Approximate literature values at room temperature, for context only
REFERENCE_GAPS = {
    "Ge": 0.67, "Si": 1.12, "InP": 1.35, "GaAs": 1.42, "CdTe": 1.5,
    "GaP": 2.26, "SiC (4H)": 3.26, "ZnO": 3.3, "GaN": 3.4, "Diamond": 5.5,
}


def classify(gap_if_nonmetal, p_nonmetal):
    """Rough material class; the boundaries are conventions, not sharp limits."""
    if p_nonmetal < 0.5:
        return "Metal / conductor"
    if gap_if_nonmetal < 3.0:
        return "Semiconductor"
    if gap_if_nonmetal < 4.0:
        return "Wide-bandgap semiconductor"
    return "Insulator"


def typical_uses(gap):
    """Typical applications of materials in this gap range."""
    if gap < 0.8:
        return "infrared detectors and thermoelectrics"
    if gap < 1.8:
        return "transistors, integrated circuits and solar cells (Si 1.1 eV, GaAs 1.4 eV)"
    if gap < 3.0:
        return "visible-light LEDs and photodetectors"
    if gap < 4.0:
        return "blue/UV LEDs and high-power electronics (GaN, SiC)"
    return "electrical insulation, dielectrics and UV-transparent windows"


bundle = get_bundle()

st.title(APP_TITLE)
st.write(
    "Predicts the experimental band gap of an inorganic compound, including "
    "semiconductors, from its chemical formula alone."
)
st.caption(f"Deployed by {AUTHOR}")

tab_predict, tab_semi, tab_model = st.tabs(
    ["Predict", "Semiconductors 101", "How the model works"]
)

# ---------------------------------------------------------------- predict
with tab_predict:
    formula = st.text_input(
        "Chemical formula", value="GaAs", help="Examples: Si, ZnO, CuInSe2, GaN"
    )

    if st.button("Predict") and formula.strip():
        try:
            r = predict_gap(formula.strip(), bundle)
        except Exception:
            st.error(
                "Could not parse that formula. Check element symbols and "
                "capitalization (e.g. 'ZnO', not 'zno')."
            )
            st.stop()

        material_class = classify(r["gap_if_nonmetal"], r["p_nonmetal"])
        st.subheader(f"{r['formula']}: {material_class}")

        c1, c2 = st.columns(2)
        c1.metric("Predicted band gap", f"{r['gap']:.2f} eV")
        c2.metric("Probability non-metal", f"{r['p_nonmetal']:.0%}")

        if r["p_nonmetal"] < 0.5:
            st.info(
                "The model classifies this compound as a likely metal, "
                "so its band gap is effectively zero."
            )
        else:
            st.write(
                f"Compounds with a gap in this range are typically used for "
                f"**{typical_uses(r['gap_if_nonmetal'])}**."
            )

        # Predicted value next to well-known semiconductors
        chart_data = dict(REFERENCE_GAPS)
        chart_data[f"{r['formula']} (predicted)"] = round(r["gap"], 2)
        chart_df = pd.Series(chart_data, name="Band gap (eV)").sort_values().to_frame()
        st.caption("Predicted gap compared with well-known semiconductors (eV)")
        st.bar_chart(chart_df)

        if r["rare_elements"]:
            st.warning(
                "Low confidence: " + ", ".join(r["rare_elements"])
                + " appear(s) rarely in the training data."
            )

        st.caption(
            "Typical error is about 0.4 eV on compositions similar to the training "
            "data and can be much larger for unfamiliar chemistry. Predictions are "
            "estimates, not a substitute for measurement."
        )

# ---------------------------------------------------------------- semiconductors
with tab_semi:
    st.header("What is a band gap?")
    st.write(
        "In a solid, electrons occupy energy bands. The **band gap** is the energy "
        "(in electron-volts, eV) an electron needs to jump from the filled valence "
        "band into the empty conduction band, where it can carry current."
    )
    st.markdown(
        """
| Material class | Band gap | Behavior |
|---|---|---|
| Metal | ~0 eV | Conducts freely |
| Semiconductor | ~0.5 to 3 eV | Conductivity can be controlled by doping, light or voltage |
| Wide-bandgap semiconductor | ~3 to 4 eV | Handles high voltage, temperature and short-wavelength light |
| Insulator | above ~4 eV | Barely conducts |
"""
    )
    st.caption("The boundaries between classes are conventions and vary between sources.")

    st.header("Why the band gap matters")
    st.markdown(
        """
- **Electronics:** the gap sets how well a transistor switches on and off. Silicon (1.1 eV) dominates chips.
- **Solar cells:** a gap near 1.3 to 1.4 eV absorbs sunlight efficiently, which is why GaAs and CdTe are used.
- **LEDs and lasers:** the gap sets the color of emitted light. GaN (3.4 eV) made blue LEDs possible.
- **Power electronics:** wide-gap materials such as SiC and GaN tolerate higher voltages and temperatures than silicon.
"""
    )

    st.header("Why predict it with machine learning?")
    st.write(
        "Measuring or computing band gaps is slow. Standard DFT calculations are "
        "expensive and systematically underestimate gaps. A trained model gives an "
        "instant estimate, which is useful for screening thousands of candidate "
        "compounds before choosing which to synthesize."
    )

# ---------------------------------------------------------------- model
with tab_model:
    st.header("Pipeline")
    st.markdown(
        """
1. **Data:** 4,604 experimentally measured band gaps (Matbench `expt_gap`, from Zhuo et al. 2018).
2. **Features:** each formula is converted into 132 numbers with **Magpie** features, which are statistics (mean, min, max, range, average deviation, mode) of elemental properties such as atomic number, electronegativity and valence electrons.
3. **Models:** Random Forest, XGBoost and a neural network (MLP) were compared on the same 80/20 split.
4. **Final model:** a two-stage XGBoost. A classifier estimates the probability that the material is a non-metal, and a regressor trained on non-metals predicts the gap. The two are multiplied.
"""
    )

    st.header("Model comparison (held-out test set)")
    comparison = pd.DataFrame(
        {
            "Model": [
                "Dummy (mean)", "Random Forest", "MLP",
                "XGBoost", "Two-stage XGBoost (soft)",
            ],
            "MAE (eV)": [1.173, 0.446, 0.479, 0.410, 0.375],
            "MAE non-metals (eV)": [1.414, 0.549, 0.588, 0.519, 0.510],
        }
    )
    st.dataframe(comparison, hide_index=True)

    st.header("Limitations")
    st.markdown(
        """
- **Unfamiliar chemistry:** when all compounds containing an element were held out of training, error rose sharply. For oxygen-containing materials it was about 1.5 eV, close to guessing the average. Elements with close chemical relatives (Ga, Zn) held up better.
- **Metal labels are noisy:** in the source dataset, zero-gap entries come largely from computed databases rather than direct measurement.
- **Composition only:** the model sees the formula, not the crystal structure, so polymorphs of the same formula get the same prediction.
- **Wide-gap materials:** there are few training examples above about 6 eV, so these predictions are less reliable.
"""
    )

# ---------------------------------------------------------------- footer
st.divider()
st.caption(f"{APP_TITLE} · Deployed by {AUTHOR}")