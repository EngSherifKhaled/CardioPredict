import streamlit as st
import joblib
import pandas as pd

# ================== Page Config ==================
st.set_page_config(
    page_title="CardioPredict",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ================== Page Header ==================
st.title("CardioPredict")
st.caption("AI-powered heart disease risk assessment • Demonstration only")

st.markdown(
    """
Predict the likelihood of heart disease using patient information.  
**Note:** This is an estimation, not a medical diagnosis.
"""
)

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalch": "Maximum Heart Rate Achieved",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Segment Slope",
    "ca": "Number of Major Vessels",
    "thal": "Thalassemia",
}

# ================== Load Model & Imputation ==================
@st.cache_resource
def load_model_and_imputers():
    model = joblib.load("model/rf_pipeline.pkl")
    impute_values = joblib.load("model/impute_values.pkl")
    median_values = impute_values["median"]
    mode_values = impute_values["mode"]
    return model, median_values, mode_values

model, median_values, mode_values = load_model_and_imputers()

# ================== Helper Functions ==================
def handle_na(value, numeric=False):
    """Convert None or 'Unknown' to pd.NA safely, convert numeric if needed."""
    if value is None or value == "Unknown" or pd.isna(value):
        return pd.NA
    return float(value) if numeric else value

def fill_missing(df):
    """Fill missing numeric/categorical values using median/mode."""
    return df.fillna({**median_values, **mode_values})

# ================== Sidebar Form ==================
with st.sidebar.form("patient_form"):
    st.header("Patient Information")

    # ----- Demographics -----
    st.subheader("Demographics")
    age_unknown = st.checkbox("Age Unknown?")
    age = None if age_unknown else st.slider(
        "Age", 20, 100, 30, help="Patient's age in years."
    )

    sex = st.selectbox(
        "Sex", ["Male", "Female"], help="Biological sex at birth."
    )

    # ----- Medical History -----
    st.subheader("Medical History")
    cp = st.selectbox(
        "Chest Pain Type",
        ["Unknown", "typical angina", "atypical angina", "non-anginal", "asymptomatic"],
        help="Type of chest pain experienced by the patient."
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar >120 mg/dL?",
        ["Unknown", "Yes", "No"],
        help="Whether fasting blood sugar is higher than 120 mg/dL (Yes=1, No=0)."
    )

    exang = st.selectbox(
        "Exercise-Induced Angina",
        ["Unknown", "Yes", "No"],
        help="Whether patient experiences chest pain during exercise."
    )

    # ----- Vital Signs & Exercise -----
    st.subheader("Vital Signs & Exercise")
    trestbps_unknown = st.checkbox("Resting BP Unknown?")
    trestbps = None if trestbps_unknown else st.slider(
        "Resting BP (mmHg)", 80, 200, 120, help="Systolic blood pressure at rest."
    )

    chol_unknown = st.checkbox("Cholesterol Unknown?")
    chol = None if chol_unknown else st.slider(
        "Cholesterol (mg/dL)", 100, 400, 200, help="Serum cholesterol level."
    )

    thalch_unknown = st.checkbox("Max Heart Rate Unknown?")
    thalch = None if thalch_unknown else st.slider(
        "Max Heart Rate Achieved", 60, 220, 150, help="Maximum heart rate during exercise."
    )

    oldpeak_unknown = st.checkbox("ST Depression Unknown?")
    oldpeak = None if oldpeak_unknown else st.slider(
        "ST Depression (oldpeak)", 0.0, 10.0, 1.0, 0.1, help="ST segment depression during exercise."
    )

    # ----- ECG & Test Results -----
    st.subheader("ECG & Test Results")
    restecg = st.selectbox(
        "Resting ECG",
        ["Unknown", "normal", "lv hypertrophy", "st-t abnormality"],
        help="Electrocardiogram results at rest."
    )

    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        ["Unknown", "upsloping", "flat", "downsloping"],
        help="Pattern of ST segment during peak exercise."
    )

    ca = st.selectbox(
        "Number of Major Vessels (0–4)",
        ["Unknown", 0, 1, 2, 3, 4],
        help="Number of vessels colored by fluoroscopy."
    )

    thal = st.selectbox(
        "Thalassemia Type",
        ["Unknown", "normal", "fixed defect", "reversable defect"],
        help="Type of thalassemia defect (blood disorder)."
    )

    submit_button = st.form_submit_button(label="Predict Risk")
    
# ================== Prediction ==================
if submit_button:
    input_dict = {
        "age": handle_na(age, numeric=True),
        "sex": 1 if sex == "Male" else 0,
        "cp": handle_na(cp),
        "trestbps": handle_na(trestbps, numeric=True),
        "chol": handle_na(chol, numeric=True),
        "fbs": 1 if fbs == "Yes" else 0 if fbs == "No" else pd.NA,
        "restecg": handle_na(restecg),
        "thalch": handle_na(thalch, numeric=True),
        "exang": 1 if exang == "Yes" else 0 if exang == "No" else pd.NA,
        "oldpeak": handle_na(oldpeak, numeric=True),
        "slope": handle_na(slope),
        "ca": handle_na(ca, numeric=True),
        "thal": handle_na(thal)
    }

    expected_columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalch", "exang", "oldpeak", "slope", "ca", "thal"
    ]

    input_data = pd.DataFrame([input_dict], columns=expected_columns)
    
    missing_cols = [col for col in input_data.columns if pd.isna(input_data[col].iloc[0])]
    input_data = fill_missing(input_data)

    # Predictions
    pred_prob = model.predict_proba(input_data)[:, 1][0]
    pred_class = int(model.predict(input_data)[0])
    pred_percent = f"{pred_prob*100:.1f}%"
    assessment = "Heart Disease Likely" if pred_class == 1 else "Heart Disease Unlikely"

    # ================== Display ==================
    st.header("Prediction Result")

    # Risk color display
    if pred_prob < 0.33:
        st.markdown(f"<div style='color:green; font-weight:bold;'>❤️ Low Risk | {pred_percent}</div>", unsafe_allow_html=True)
    elif pred_prob < 0.66:
        st.markdown(f"<div style='color:orange; font-weight:bold;'>⚠️ Moderate Risk | {pred_percent}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='color:red; font-weight:bold;'>❌ High Risk | {pred_percent}</div>", unsafe_allow_html=True)

    # Missing data info
    st.markdown("---")
    if missing_cols:
        with st.expander("Note: Some input values were missing"):
            st.markdown(
                f"**Missing inputs:** {', '.join(missing_cols)}<br>"
                "Missing values were handled using statistical imputation "
                "(median for numeric features, most frequent value for categorical features).<br>"
                "Providing all inputs improves prediction reliability.<br>"
                "This tool provides risk estimation only and is not a medical diagnosis.",
                unsafe_allow_html=True
                )

# ================== About Panel ==================
with st.expander("About this Model"):
    st.markdown("""
- **Model:** Random Forest with preprocessing pipeline  
- **Input Features:** Age, Sex, Resting BP, Cholesterol, Chest Pain Type, Fasting Blood Sugar, ECG, Max Heart Rate, Exercise Angina, ST Depression, Slope, Number of Major Vessels, Thalassemia  
- **Missing Data Handling:** Median imputation for numeric, Mode imputation for categorical  
- **Output:** Predicted risk probability and class (Heart Disease Likely / Unlikely)  
- **Limitations:** Risk estimation only, not a medical diagnosis. Should not replace clinical evaluation.  
- **Reference:** Original dataset used for training (e.g., UCI Heart Disease dataset)
""")
