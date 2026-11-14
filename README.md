# Machine Learning Band Gap Predictor for Semiconductors & Inorganic Materials

Machine learning model that predicts the **experimental band gap** of an inorganic compound, including semiconductors, from its chemical formula alone, deployed as a Streamlit web app.

**Live app:** YOUR-STREAMLIT-LINK

Deployed by Hamid Kofi Abdul

## Motivation

The band gap determines whether a material behaves as a metal, semiconductor or insulator, and it sets the suitability of a material for transistors, solar cells, LEDs and power electronics. Measuring gaps is slow, and standard DFT calculations are expensive and systematically underestimate them. A trained model gives an instant estimate that can be used to screen candidate compounds before synthesis.

## Dataset

[Matbench](https://matbench.materialsproject.org/) `matbench_expt_gap`: 4,604 compositions with experimentally measured band gaps, sourced from Zhuo et al. (2018).

- 53.2% of entries have a gap of exactly 0 eV (metals).
- No duplicate formulas.
- Caveat: only the non-zero gaps come from experimental literature. The zero-gap entries largely come from computed databases, so the metal/non-metal labels are noisier than the gap values.

## Method

1. **Features:** each formula is converted into 132 numbers with [Magpie](https://doi.org/10.1038/npjcompumats.2016.28) composition features (mean, min, max, range, average deviation and mode of 22 elemental properties), computed with `matminer`.
2. **Models compared** on one shared 80/20 split: Random Forest, XGBoost and an MLP (with quantile-transformed inputs).
3. **Final model:** a two-stage XGBoost. A classifier estimates the probability that the material is a non-metal, and a regressor trained only on non-metals predicts the gap. The prediction is the product of the two (the "soft" version), which avoids the large errors of a hard metal/non-metal cutoff.

## Results

Held-out test set (921 materials):

| Model | MAE (eV) | MAE non-metals (eV) | RMSE (eV) | R² | Metal accuracy |
|---|---|---|---|---|---|
| Dummy (training mean) | 1.173 | 1.414 | 1.467 | 0.000 | 0.450 |
| Random Forest | 0.446 | 0.549 | 0.842 | 0.670 | 0.765 |
| MLP | 0.479 | 0.588 | 0.865 | 0.652 | 0.731 |
| XGBoost | 0.410 | 0.519 | 0.796 | 0.705 | 0.797 |
| Two-stage XGBoost (hard) | 0.366 | 0.522 | 0.930 | 0.598 | 0.911 |
| **Two-stage XGBoost (soft)** | 0.375 | **0.510** | 0.836 | 0.675 | 0.860 |

The soft two-stage model was chosen for deployment: its overall MAE is within 0.01 eV of the hard version, and it has the best non-metals MAE and a much lower RMSE.

### Generalization to unfamiliar chemistry

A random split is optimistic because similar compounds appear in both train and test. To measure generalization, all compounds containing one element were held out of training and used as the test set (single-stage XGBoost):

| Held-out element | Test size | MAE (eV) | Dummy MAE (eV) |
|---|---|---|---|
| O | 1090 | 1.508 | 1.647 |
| S | 652 | 0.794 | 1.215 |
| Se | 676 | 0.609 | 1.005 |
| Cu | 390 | 0.625 | 0.926 |
| Zn | 213 | 0.498 | 1.220 |
| Ga | 371 | 0.398 | 1.079 |

Error stays close to the random-split score for elements with close chemical relatives in the data (Ga, Zn) and is much worse for oxygen, which has no close substitute.

## Limitations

- **Unfamiliar chemistry:** realistic error on compositions unlike the training data is closer to 0.5 to 0.8 eV and can be larger.
- **Composition only:** the model does not see crystal structure, so polymorphs of the same formula receive the same prediction.
- **Wide-gap materials:** few training examples above about 6 eV, so these predictions are less reliable.
- **Label noise:** metal labels in the source data are partly computational.
- **Scope:** the dataset covers inorganic compounds in general (metals, semiconductors and insulators). The model does not account for doping, defects, temperature or device performance.
- Predictions are estimates and not a substitute for measurement or DFT.

## Project structure

```
Semiconductor-BandGap-Prediction/
├── app.py                    # Streamlit app
├── predict.py                # Featurization and prediction logic
├── requirements.txt          # Pinned dependencies
├── models/
│   └── bandgap_model.joblib  # Trained two-stage model and metadata
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── data/raw/                 # Raw dataset files
└── src/
```

## Run locally

```bash
git clone https://github.com/YOUR-USERNAME/Semiconductor-BandGap-Prediction.git
cd Semiconductor-BandGap-Prediction

python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Reproduce the results

Open `notebooks/01_eda_and_modeling.ipynb` and run all cells. The notebook loads the data, builds the features, trains the three models, runs the evaluation and chemistry hold-out tests, and saves the final model to `models/bandgap_model.joblib`.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, matminer, pymatgen, Streamlit.

## References

- Y. Zhuo, A. Mansouri Tehrani, J. Brgoch. *Predicting the Band Gaps of Inorganic Solids by Machine Learning.* J. Phys. Chem. Lett. 2018, 9, 1668-1673.
- A. Dunn, Q. Wang, A. Ganose, D. Dopp, A. Jain. *Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm.* npj Comput. Mater. 2020.
- L. Ward, A. Agrawal, A. Choudhary, C. Wolverton. *A general-purpose machine learning framework for predicting properties of inorganic materials.* npj Comput. Mater. 2016.
- L. Ward et al. *Matminer: An open source toolkit for materials data mining.* Comput. Mater. Sci. 2018.

## Author
Hamid Kofi Abdul


