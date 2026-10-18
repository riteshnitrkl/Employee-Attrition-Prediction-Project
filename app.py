from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "employee_attrition.csv"
MODEL_PATH = PROJECT_DIR / "top_8_LogisticRegression_Tuned.joblib"

st.set_page_config(page_title="Employee Attrition Risk Predictor", page_icon="📊", layout="wide")

@st.cache_resource
def load_model_bundle(path):
    bundle = joblib.load(path)
    if isinstance(bundle, dict) and "pipeline" in bundle:
        return bundle
    return {"pipeline": bundle, "features": None, "model_name": "Unknown"}

@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)

st.title("📊 Employee Attrition Risk Predictor")
st.write(
    "Enter employee information to estimate the model's predicted probability "
    "of attrition. This is a decision-support estimate, not a certainty."
)

missing = [str(p) for p in (DATA_PATH, MODEL_PATH) if not p.exists()]
if missing:
    st.error("Required file(s) not found:")
    for item in missing:
        st.code(item)
    st.info("Keep app.py in the Hexaware_OA project root, with data/ and saved_models/ beside it.")
    st.stop()

try:
    bundle = load_model_bundle(str(MODEL_PATH))
    raw_df = load_dataset(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load model or dataset: {exc}")
    st.stop()

pipeline = bundle["pipeline"]
features = bundle.get("features")
if not features:
    st.error("The saved model bundle has no feature list. Rerun Step 20 to save the tuned model bundle.")
    st.stop()

missing_features = [c for c in features if c not in raw_df.columns]
if missing_features:
    st.error(f"Model features missing from dataset: {missing_features}")
    st.stop()

schema = raw_df[features].copy()

with st.sidebar:
    st.header("Prediction settings")
    threshold = st.slider(
        "Classification threshold", 0.05, 0.95, 0.50, 0.05,
        help="Changes the class label only. Validate a business threshold before using it."
    )
    st.caption("Model: top_8_LogisticRegression_Tuned")

st.subheader("Employee details")
st.caption("Complete the fields and click Predict. The form uses the eight features saved with the selected model.")

with st.form("employee_form"):
    values = {}
    columns = st.columns(2)

    for i, feature in enumerate(features):
        series = schema[feature]
        with columns[i % 2]:
            label = feature.replace("_", " ").title()

            if pd.api.types.is_bool_dtype(series):
                values[feature] = st.selectbox(label, [False, True])
            elif pd.api.types.is_numeric_dtype(series):
                nums = pd.to_numeric(series, errors="coerce").dropna()
                default = float(nums.median()) if not nums.empty else 0.0
                values[feature] = st.number_input(label, value=default)
            else:
                choices = sorted(series.dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist())
                if not choices:
                    choices = ["Unknown"]
                mode = series.dropna().astype(str).mode()
                default_index = choices.index(mode.iloc[0]) if not mode.empty and mode.iloc[0] in choices else 0
                values[feature] = st.selectbox(label, choices, index=default_index)

    submitted = st.form_submit_button("Predict attrition risk", type="primary")

if submitted:
    input_df = pd.DataFrame([values], columns=features)
    for feature in features:
        source = schema[feature]
        if pd.api.types.is_numeric_dtype(source):
            input_df[feature] = pd.to_numeric(input_df[feature], errors="coerce")
        elif pd.api.types.is_bool_dtype(source):
            input_df[feature] = input_df[feature].astype(bool)

    try:
        probabilities = pipeline.predict_proba(input_df)
        classes = list(pipeline.named_steps["model"].classes_)
        positive_index = classes.index(1) if 1 in classes else len(classes) - 1
        probability = float(probabilities[0, positive_index])
        prediction = probability >= threshold

        st.divider()
        st.subheader("Prediction")
        left, right = st.columns(2)
        left.metric("Predicted attrition probability", f"{probability:.1%}")
        right.metric("Classification", "Higher risk" if prediction else "Lower risk")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"Threshold: {threshold:.0%}. The probability is an estimate, not a guarantee.")
        with st.expander("View submitted inputs"):
            st.dataframe(input_df, use_container_width=True)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.exception(exc)

st.divider()
st.caption("Model and dataset are for demonstration purposes only. They are not intended for production use or to make real-world decisions."
)
