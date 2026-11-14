"""Band gap prediction from chemical composition (two-stage XGBoost on Magpie features)."""
import joblib
import numpy as np
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

_magpie = ElementProperty.from_preset("magpie")


def load_bundle(path="models/bandgap_model.joblib"):
    return joblib.load(path)


def predict_gap(formula, bundle, rare_threshold=20):
    """Return predicted gap (eV), P(non-metal) and low-confidence flags.

    The prediction is P(non-metal) multiplied by the gap predicted by a
    regressor trained on non-metals. Raises on unparseable formulas.
    """
    comp = Composition(formula)
    x = np.array(_magpie.featurize(comp)).reshape(1, -1)

    p_nonmetal = float(bundle["classifier"].predict_proba(x)[0, 1])
    gap_if_nonmetal = float(
        np.clip(bundle["regressor"].predict(x)[0], 0, bundle["max_gap"])
    )

    # Elements with few training examples are flagged as low-confidence
    counts = bundle["element_counts"]
    rare = [el for el in comp.get_el_amt_dict() if counts.get(el, 0) < rare_threshold]

    return {
        "formula": comp.reduced_formula,
        "gap": p_nonmetal * gap_if_nonmetal,
        "gap_if_nonmetal": gap_if_nonmetal,
        "p_nonmetal": p_nonmetal,
        "rare_elements": rare,
    }