"""Streamlit frontend for the Ames house-price polynomial regression model.

Run:  streamlit run app.py
Place this file next to house_price_model.joblib (or change MODEL_PATH).
Use the same virtual environment / scikit-learn version the model was saved with.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "house_price_model.joblib"

# Typical error of the model: validation RMSE from 5-fold CV (degree 2, OverallQual in poly group)
CV_RMSE = 24_742

# Numeric inputs: (min, max, default, help). Ranges are those of the cleaned training data
# (after removing the 2 GrLivArea anomalies and the 4 LotArea > 100,000 records).
# Adjust them if your own min/max differ: train_data[col].agg(["min", "max"])
NUMERIC = {
    "LotArea":     (1_300, 70_800, 9_600, "Lot size in square feet"),
    "YearBuilt":   (1872, 2010, 1995, "Construction year"),
    "MasVnrArea":  (0, 1_600, 0, "Masonry veneer area in square feet (0 = none)"),
    "TotalBsmtSF": (0, 3_210, 1_000, "Total basement area in square feet (0 = no basement)"),
    "GrLivArea":   (334, 4_480, 1_600, "Above-ground living area in square feet"),
    "FullBath":    (0, 3, 2, "Full bathrooms above grade"),
    "Fireplaces":  (0, 3, 1, "Number of fireplaces"),
    "GarageCars":  (0, 4, 2, "Garage capacity in cars (0 = no garage)"),
    "OverallQual": (1, 10, 7, "Overall material and finish quality (1 = very poor, 10 = very excellent)"),
}

# Categorical inputs: (options, default). Ordinal options are ordered low -> high.
BSMT_EXPOSURE = ["None", "No", "Mn", "Av", "Gd"]
BSMT_FIN_TYPE = ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]
FIREPLACE_QU = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
GARAGE_FINISH = ["None", "Unf", "RFn", "Fin"]
HEATING_QC = ["Po", "Fa", "TA", "Gd", "Ex"]
NEIGHBORHOODS = ["Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor",
                 "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel", "NAmes", "NPkVill",
                 "NWAmes", "NoRidge", "NridgHt", "OldTown", "SWISU", "Sawyer", "SawyerW",
                 "Somerst", "StoneBr", "Timber", "Veenker"]
LOT_CONFIG = ["Inside", "Corner", "CulDSac", "FR2", "FR3"]
BLDG_TYPE = ["1Fam", "2fmCon", "Duplex", "Twnhs", "TwnhsE"]
SALE_CONDITION = ["Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"]


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


def num_input(col: str, widget: str = "number"):
    lo, hi, default, help_text = NUMERIC[col]
    help_text = f"{help_text}. Training range: {lo:,} to {hi:,}."
    if widget == "slider":
        return st.slider(col, lo, hi, default, help=help_text)
    return st.number_input(col, min_value=lo, max_value=hi, value=default, step=1, help=help_text)


def select(label: str, options: list, default: str, help_text: str = ""):
    return st.selectbox(label, options, index=options.index(default), help=help_text or None)


st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 House Price Predictor")
st.caption("Polynomial regression (degree 2) on the Ames housing data. "
           "Enter raw values; the saved pipeline handles imputation, encoding, polynomial expansion and scaling.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}. Put house_price_model.joblib next to app.py.")
    st.stop()
model = load_model(MODEL_PATH)

left, mid, right = st.columns(3)

with left:
    st.subheader("Size and lot")
    LotArea = num_input("LotArea")
    GrLivArea = num_input("GrLivArea")
    TotalBsmtSF = num_input("TotalBsmtSF")
    MasVnrArea = num_input("MasVnrArea")
    FullBath = num_input("FullBath", "slider")

with mid:
    st.subheader("Quality and features")
    OverallQual = num_input("OverallQual", "slider")
    YearBuilt = num_input("YearBuilt")
    HeatingQC = select("HeatingQC", HEATING_QC, "Ex", "Heating quality and condition")
    CentralAir = select("CentralAir", ["Y", "N"], "Y", "Central air conditioning")
    Fireplaces = num_input("Fireplaces", "slider")
    FireplaceQu = select("FireplaceQu", FIREPLACE_QU, "TA", "Fireplace quality ('None' = no fireplace)")
    GarageCars = num_input("GarageCars", "slider")
    GarageFinish = select("GarageFinish", GARAGE_FINISH, "RFn", "Interior garage finish ('None' = no garage)")

with right:
    st.subheader("Basement, location and sale")
    BsmtExposure = select("BsmtExposure", BSMT_EXPOSURE, "No", "Walkout/garden-level exposure ('None' = no basement)")
    BsmtFinType1 = select("BsmtFinType1", BSMT_FIN_TYPE, "GLQ", "Basement finished area rating ('None' = no basement)")
    Neighborhood = select("Neighborhood", NEIGHBORHOODS, "NAmes")
    BldgType = select("BldgType", BLDG_TYPE, "1Fam", "Dwelling type")
    LotConfig = select("LotConfig", LOT_CONFIG, "Inside")
    SaleCondition = select("SaleCondition", SALE_CONDITION, "Normal")

# Consistency checks between related inputs (the model was trained on consistent records)
warnings = []
if TotalBsmtSF == 0 and (BsmtExposure != "None" or BsmtFinType1 != "None"):
    warnings.append("TotalBsmtSF is 0, so BsmtExposure and BsmtFinType1 should be 'None'.")
if TotalBsmtSF > 0 and (BsmtExposure == "None" or BsmtFinType1 == "None"):
    warnings.append("TotalBsmtSF is above 0, so BsmtExposure and BsmtFinType1 should not be 'None'.")
if Fireplaces == 0 and FireplaceQu != "None":
    warnings.append("Fireplaces is 0, so FireplaceQu should be 'None'.")
if Fireplaces > 0 and FireplaceQu == "None":
    warnings.append("Fireplaces is above 0, so FireplaceQu should not be 'None'.")
if GarageCars == 0 and GarageFinish != "None":
    warnings.append("GarageCars is 0, so GarageFinish should be 'None'.")
if GarageCars > 0 and GarageFinish == "None":
    warnings.append("GarageCars is above 0, so GarageFinish should not be 'None'.")
for w in warnings:
    st.warning(w)

if st.button("Predict price", type="primary"):
    row = pd.DataFrame([{
        "LotArea": LotArea, "LotConfig": LotConfig, "Neighborhood": Neighborhood,
        "BldgType": BldgType, "OverallQual": OverallQual, "YearBuilt": YearBuilt,
        "MasVnrArea": MasVnrArea, "BsmtExposure": BsmtExposure, "BsmtFinType1": BsmtFinType1,
        "TotalBsmtSF": TotalBsmtSF, "HeatingQC": HeatingQC, "CentralAir": CentralAir,
        "GrLivArea": GrLivArea, "FullBath": FullBath, "Fireplaces": Fireplaces,
        "FireplaceQu": FireplaceQu, "GarageFinish": GarageFinish, "GarageCars": GarageCars,
        "SaleCondition": SaleCondition,
    }])
    row = row[list(model.feature_names_in_)]  # same column order as training

    price = float(model.predict(row)[0])
    st.metric("Predicted sale price", f"${price:,.0f}")
    st.caption(f"Typical error of this model (cross-validated RMSE): about ±${CV_RMSE:,.0f}, "
               f"so a plausible range is ${max(price - CV_RMSE, 0):,.0f} to ${price + CV_RMSE:,.0f}.")
    with st.expander("Input sent to the model"):
        st.dataframe(row.T.astype(str).rename(columns={0: "value"}), width="stretch")
