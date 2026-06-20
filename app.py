import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .stApp { font-family: 'Segoe UI', sans-serif; }

    h1 {
        color: #1b2a4a;
        font-weight: 700;
        padding-bottom: 0px;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-top: -10px;
        margin-bottom: 25px;
    }

    /* Risk cards */
    .risk-card {
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 15px;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #c0392b 100%);
        color: white;
    }
    .risk-low {
        background: linear-gradient(135deg, #2ecc71 0%, #1b9e6b 100%);
        color: white;
    }
    .risk-percent {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
    }
    .risk-label {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 5px;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background-color: #1b2a4a;
    }

    /* Default: sidebar text (labels, captions, headers) is light */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #f1f3f6 !important;
    }

    /* Override: text INSIDE the actual input boxes (selectbox, dropdown
       options, number/slider value boxes) must stay dark, since those
       boxes render with a white/light background */
    section[data-testid="stSidebar"] [data-baseweb="select"] * ,
    section[data-testid="stSidebar"] [data-baseweb="popover"] * ,
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] [role="listbox"] * ,
    section[data-testid="stSidebar"] [role="option"] {
        color: #1b2a4a !important;
    }

    /* Expander headers — works across current Streamlit versions */
    section[data-testid="stSidebar"] details {
        background-color: #2a3b5e !important;
        border-radius: 8px !important;
        border: none !important;
    }
    section[data-testid="stSidebar"] details summary {
        background-color: #2a3b5e !important;
        border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] details summary span,
    section[data-testid="stSidebar"] details summary p,
    section[data-testid="stSidebar"] details summary svg {
        color: #f1f3f6 !important;
        fill: #f1f3f6 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: #2a3b5e !important;
        border-radius: 8px !important;
        margin-bottom: 8px;
    }

    /* Metric boxes */
    div[data-testid="stMetric"] {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }

    .info-box {
        background-color: #eef2ff;
        border-left: 4px solid #4f6df5;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    lr_model = joblib.load('logistic_regression_model.pkl')
    svm_model = joblib.load('svm_rbf_model.pkl')
    scaler = joblib.load('robust_scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    numeric_features = joblib.load('numeric_features.pkl')
    lr_explainer = joblib.load('lr_shap_explainer.pkl')
    svm_explainer = joblib.load('svm_shap_explainer.pkl')
    background = joblib.load('shap_background.pkl')
    return (lr_model, svm_model, scaler, feature_columns, numeric_features,
            lr_explainer, svm_explainer, background)

(lr_model, svm_model, scaler, feature_columns, numeric_features,
 lr_explainer, svm_explainer, background) = load_artifacts()

# ============================================================
# SAMPLE PATIENTS (for "Load Sample" buttons)
# ============================================================
SAMPLE_PATIENTS = {
    "Low-risk example": dict(
        age=42, sex="Female", cp="non-anginal", trestbps=118, chol=190,
        fbs="No", restecg="normal", thalch=172, exang="No", oldpeak=0.2,
        slope="upsloping", ca=0, thal="normal"
    ),
    "High-risk example": dict(
        age=64, sex="Male", cp="asymptomatic", trestbps=152, chol=286,
        fbs="Yes", restecg="lv hypertrophy", thalch=108, exang="Yes", oldpeak=2.6,
        slope="flat", ca=2, thal="reversable defect"
    ),
    "Incomplete record example": dict(
        age=58, sex="Male", cp="atypical angina", trestbps=130, chol=245,
        fbs="No", restecg="normal", thalch=150, exang="No", oldpeak=1.0,
        slope=None, ca=None, thal=None
    ),
}

if "loaded_patient" not in st.session_state:
    st.session_state.loaded_patient = SAMPLE_PATIENTS["Low-risk example"]

# ============================================================
# SIDEBAR — INPUT FORM
# ============================================================
with st.sidebar:
    st.markdown("## 🩺 Patient Information")
    st.caption("Fill in known clinical values. Tests not performed can be left blank.")

    st.markdown("#### Try a sample patient")
    sample_choice = st.selectbox(
        "Quick-load example", ["— Select —"] + list(SAMPLE_PATIENTS.keys()),
        label_visibility="collapsed"
    )
    if sample_choice != "— Select —":
        st.session_state.loaded_patient = SAMPLE_PATIENTS[sample_choice]

    p = st.session_state.loaded_patient
    st.divider()

    with st.expander("👤 Basic Information", expanded=True):
        age = st.slider("Age", 18, 100, p["age"],
                         help="Patient's age in years.")
        sex = st.radio("Sex", ["Male", "Female"],
                        index=0 if p["sex"] == "Male" else 1, horizontal=True)

    with st.expander("🫀 Chest Pain & Vitals", expanded=True):
        cp = st.selectbox(
            "Chest Pain Type",
            ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
            index=["typical angina", "atypical angina", "non-anginal", "asymptomatic"].index(p["cp"]),
            help=("Typical angina = classic exertional chest pain. "
                  "Asymptomatic = no chest pain reported, despite possible underlying disease.")
        )
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, p["trestbps"],
                              help="Blood pressure on admission to hospital. Normal resting ~120.")
        chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, p["chol"],
                          help="Total serum cholesterol. Desirable level is under 200 mg/dl.")
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"],
                        index=0 if p["fbs"] == "No" else 1, horizontal=True,
                        help="Whether fasting blood sugar exceeds 120 mg/dl (a diabetes indicator).")

    with st.expander("📈 Exercise / ECG Results", expanded=True):
        restecg = st.selectbox(
            "Resting ECG Results",
            ["normal", "lv hypertrophy", "st-t abnormality"],
            index=["normal", "lv hypertrophy", "st-t abnormality"].index(p["restecg"]),
            help="Result of resting electrocardiogram."
        )
        thalch = st.slider("Max Heart Rate Achieved", 60, 220, p["thalch"],
                            help="Highest heart rate reached during exercise stress test.")
        exang = st.radio("Exercise-Induced Angina?", ["No", "Yes"],
                          index=0 if p["exang"] == "No" else 1, horizontal=True,
                          help="Whether chest pain occurred during exercise.")
        oldpeak = st.slider("ST Depression (oldpeak)", -2.0, 6.5, float(p["oldpeak"]), step=0.1,
                             help="ST depression induced by exercise relative to rest — a key ECG stress marker.")

    with st.expander("🔬 Advanced Tests (optional)", expanded=True):
        st.markdown(
            '<div class="info-box">These tests (angiogram / thallium scan) are not always performed. '
            'Leave blank if unavailable — the model accounts for missing results.</div>',
            unsafe_allow_html=True
        )
        slope_options = ["upsloping", "flat", "downsloping"]
        slope = st.selectbox("Slope of Peak Exercise ST Segment", slope_options,
                              index=slope_options.index(p["slope"]) if p["slope"] in slope_options else None,
                              placeholder="Not tested / unknown")
        ca = st.selectbox("Number of Major Vessels Colored (0–3)", [0, 1, 2, 3],
                           index=[0, 1, 2, 3].index(p["ca"]) if p["ca"] in [0, 1, 2, 3] else None,
                           placeholder="Not tested / unknown")
        thal_options = ["normal", "fixed defect", "reversable defect"]
        thal = st.selectbox("Thalassemia Test Result", thal_options,
                             index=thal_options.index(p["thal"]) if p["thal"] in thal_options else None,
                             placeholder="Not tested / unknown")

    st.divider()
    st.markdown("#### 🤖 Model Selection")
    model_choice = st.radio(
        "Choose prediction model",
        ["SVM (RBF) — Higher Recall ✅ Recommended", "Logistic Regression — More Interpretable"],
        help="SVM catches more true disease cases (higher recall). Logistic Regression is faster to explain."
    )

    predict_button = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

# ============================================================
# BUILD INPUT ROW (matches training pipeline exactly)
# ============================================================
def build_input_row(age, sex, trestbps, chol, fbs, restecg, thalch, exang,
                     oldpeak, cp, slope, ca, thal, feature_columns, numeric_features):
    row = {col: 0 for col in feature_columns}

    row['age'] = age
    row['sex'] = 1 if sex == "Male" else 0
    row['trestbps'] = trestbps
    row['chol'] = chol
    row['fbs'] = 1 if fbs == "Yes" else 0
    row['thalch'] = thalch
    row['exang'] = 1 if exang == "Yes" else 0
    row['oldpeak'] = oldpeak

    row['ca_missing'] = 1 if ca is None else 0
    row['ca'] = 0 if ca is None else ca

    row['slope_missing'] = 1 if slope is None else 0
    row['thal_missing'] = 1 if thal is None else 0

    cp_col = f'cp_{cp}'
    if cp_col in row:
        row[cp_col] = True

    restecg_col = f'restecg_{restecg}'
    if restecg_col in row:
        row[restecg_col] = True

    if slope is not None:
        slope_col = f'slope_{slope}'
        if slope_col in row:
            row[slope_col] = True

    if thal is not None:
        thal_col = f'thal_{thal}'
        if thal_col in row:
            row[thal_col] = True

    input_df = pd.DataFrame([row])[feature_columns]
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])
    return input_df

# ============================================================
# MAIN PAGE — HEADER
# ============================================================
st.markdown("# ❤️ Heart Disease Risk Predictor")
st.markdown(
    '<p class="subtitle">An interpretable machine learning dashboard for estimating '
    'heart disease risk from clinical attributes — built on 920 patient records '
    'across 4 hospital sites.</p>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Model Insights", "ℹ️ About This Project"])

# ------------------------------------------------------------
# TAB 1 — PREDICTION
# ------------------------------------------------------------
with tab1:
    if predict_button:
        input_df = build_input_row(age, sex, trestbps, chol, fbs, restecg, thalch, exang,
                                    oldpeak, cp, slope, ca, thal, feature_columns, numeric_features)

        use_svm = model_choice.startswith("SVM")
        model = svm_model if use_svm else lr_model
        model_name = "SVM (RBF)" if use_svm else "Logistic Regression"

        proba = model.predict_proba(input_df)[0][1]
        is_high_risk = proba >= 0.5

        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            risk_class = "risk-high" if is_high_risk else "risk-low"
            risk_label = "⚠️ Disease Likely" if is_high_risk else "✅ Disease Unlikely"
            st.markdown(f"""
                <div class="risk-card {risk_class}">
                    <p class="risk-percent">{proba:.0%}</p>
                    <p class="risk-label">{risk_label}</p>
                </div>
            """, unsafe_allow_html=True)

            st.metric("Model Used", model_name)
            st.progress(float(proba))
            st.caption(f"Predicted probability of heart disease: **{proba:.1%}**")

            with st.expander("📋 Patient Summary"):
                st.write(f"**Age:** {age}  |  **Sex:** {sex}")
                st.write(f"**Chest Pain:** {cp}  |  **Resting BP:** {trestbps} mmHg")
                st.write(f"**Cholesterol:** {chol} mg/dl  |  **Max HR:** {thalch}")
                st.write(f"**Exercise Angina:** {exang}  |  **Oldpeak:** {oldpeak}")
                st.write(f"**Slope:** {slope or 'Not tested'}  |  **Ca:** {ca if ca is not None else 'Not tested'}  |  **Thal:** {thal or 'Not tested'}")

        with col2:
            st.subheader("🧠 Why this prediction?")
            st.caption("SHAP values show how each factor pushed the prediction up (red) or down (blue).")

            if use_svm:
                shap_vals = svm_explainer.shap_values(input_df, nsamples=100)
                shap_vals_disease = shap_vals[:, :, 1] if shap_vals.ndim == 3 else shap_vals[1]
                base_val = svm_explainer.expected_value[1]
            else:
                shap_vals_disease = lr_explainer.shap_values(input_df)
                base_val = lr_explainer.expected_value

            fig, ax = plt.subplots(figsize=(8, 5))
            shap.waterfall_plot(shap.Explanation(
                values=shap_vals_disease[0], base_values=base_val,
                data=input_df.iloc[0], feature_names=feature_columns
            ), show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.warning("⚠️ This tool is for educational purposes only and is not a substitute for professional medical diagnosis.")

    else:
        st.info("👈 Fill in the patient details in the sidebar (or load a sample patient) and click **Predict Risk**.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Patients in Training Data", "920")
        c2.metric("Best Model ROC-AUC", "0.91")
        c3.metric("Clinical Features Used", "13")

# ------------------------------------------------------------
# TAB 2 — MODEL INSIGHTS
# ------------------------------------------------------------
with tab2:
    st.subheader("Model Comparison")
    comparison_data = pd.DataFrame({
        "Model": ["SVM (RBF) — Tuned", "Logistic Regression — Tuned", "Random Forest",
                  "Deep Learning (MLP)", "Gradient Boosting", "XGBoost", "KNN"],
        "Accuracy": [0.821, 0.832, 0.804, 0.821, 0.826, 0.821, 0.799],
        "Recall": [0.922, 0.892, 0.863, 0.882, 0.902, 0.892, 0.863],
        "ROC-AUC": [0.909, 0.907, 0.907, 0.904, 0.902, 0.896, 0.891],
    }).sort_values("ROC-AUC", ascending=False)

    st.dataframe(comparison_data, use_container_width=True, hide_index=True)
    st.bar_chart(comparison_data.set_index("Model")[["Accuracy", "Recall", "ROC-AUC"]])

    st.subheader("Why SVM (RBF) Was Chosen")
    st.markdown("""
    - **Highest Recall (0.922)** — misses fewer true disease cases, which matters more
      than raw accuracy in a screening context.
    - **Highest ROC-AUC** — best overall ranking of patient risk.
    - SHAP analysis showed SVM relies more on direct clinical measurements rather than
      missingness patterns tied to hospital site — suggesting better generalization.
    """)

# ------------------------------------------------------------
# TAB 3 — ABOUT
# ------------------------------------------------------------
with tab3:
    st.subheader("About This Project")
    st.markdown("""
    This dashboard is the deployment layer of a heart disease prediction project built on
    the multi-site UCI heart disease dataset (Cleveland, Hungary, Switzerland, VA Long Beach).

    **Pipeline:** data cleaning (disguised missing values, MNAR imputation) → feature
    selection (ANOVA, Chi-square, model-based importance) → training & comparison of 7
    models → SHAP interpretability → hyperparameter tuning → deployment.

    **Disclaimer:** This tool was built for academic and portfolio purposes. It is not a
    certified medical device and should never be used for real clinical decision-making.
    """)