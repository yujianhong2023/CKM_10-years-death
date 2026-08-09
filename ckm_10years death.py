import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="CKM 10-Year Mortality Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.0rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #6c757d;
        margin-bottom: 1.2rem;
    }
    .divider {
        border-top: 1px solid #e9ecef;
        margin: 1.2rem 0;
    }

    .metric-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-box .label {
        font-size: 0.65rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .metric-box .value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
    }

    .input-card {
        background-color: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #e9ecef;
        margin-bottom: 0.8rem;
    }
    .input-card-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.8rem;
    }

    .result-card {
        border-radius: 12px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        margin-bottom: 0.8rem;
        border: 2px solid #e9ecef;
        background: linear-gradient(135deg, #fafbfc, #ffffff);
    }
    .result-number {
        font-size: 3.5rem;
        font-weight: 700;
        color: #1a1a2e;
        line-height: 1.2;
    }
    .result-number .percent {
        font-size: 1.8rem;
        color: #6c757d;
    }
    .result-label {
        font-size: 0.95rem;
        color: #6c757d;
        margin-top: 0.2rem;
    }
    .result-prob-detail {
        margin-top: 0.4rem;
        font-size: 0.8rem;
        color: #6c757d;
    }

    .probability-bar {
        margin-top: 0.8rem;
        background-color: #e9ecef;
        border-radius: 20px;
        height: 12px;
        overflow: hidden;
        position: relative;
    }
    .probability-bar .death-bar {
        background: linear-gradient(90deg, #dc3545, #b02a37);
        height: 100%;
        border-radius: 20px;
        transition: width 0.5s ease;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4a6cf7, #6a3de8);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.6rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74, 108, 247, 0.4);
    }

    .placeholder {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 2.5rem 1.5rem;
        text-align: center;
        border: 2px dashed #dee2e6;
    }
    .placeholder-icon {
        font-size: 2.5rem;
        margin: 0;
    }
    .placeholder-text {
        font-size: 0.95rem;
        color: #6c757d;
        margin-top: 0.3rem;
    }

    .interpret-box {
        border-radius: 8px;
        padding: 0.7rem 1rem;
        border-left: 4px solid;
    }
    .interpret-death {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .interpret-survive {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .interpret-title {
        font-weight: 600;
        margin: 0;
    }
    .interpret-text {
        font-size: 0.85rem;
        margin: 0.2rem 0 0 0;
    }

    .info-note {
        text-align: center;
        font-size: 0.75rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
    .threshold-note {
        text-align: center;
        color: #6c757d;
        font-size: 0.75rem;
        margin-bottom: 0.8rem;
        padding: 0.3rem;
        background-color: #f8f9fa;
        border-radius: 6px;
    }

    .feature-importance-note {
        text-align: center;
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 0.5rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


# ==================== Load Model ====================
@st.cache_resource
def load_model():
    """Load trained model and preprocessors for 10-year mortality"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    possible_paths = [
        os.path.join(current_dir, 'ckm_risk_model_10yr.pkl'),
        r"C:\Users\admin\PycharmProjects\PythonProject9\CKM_10_year_death\ckm_risk_model_10yr.pkl",
        r"C:\Users\admin\PycharmProjects\PythonProject9\ckm_risk_model_10yr.pkl"
    ]

    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as file:
                    artifacts = pickle.load(file)
                return artifacts
            except Exception as e:
                continue

    st.error("❌ Model file 'ckm_risk_model_10yr.pkl' not found. Please ensure the model file exists in the correct path.")
    st.info("💡 Expected paths checked:\n" + "\n".join(possible_paths))
    return None


artifacts = load_model()

if artifacts is None:
    st.stop()

model = artifacts['model']
scaler = artifacts['scaler']
features = artifacts['features']
categorical_features = artifacts['categorical_features']
continuous_features = artifacts['continuous_features']
model_threshold = artifacts.get('threshold', 0.5)

# ===== IMPORTANT: Override threshold to 0.50 for binary classification display =====
DISPLAY_THRESHOLD = 0.50

model_info = artifacts.get('model_info', {})

# ==================== Header ====================
st.markdown('<p class="main-header">🏥 CKM 10-Year All-Cause Mortality Risk Prediction</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Cardiovascular-Kidney-Metabolic Syndrome Risk Assessment Tool (Extended 10-Year Model with 29 Clinical Features)</p>',
    unsafe_allow_html=True)

# ==================== Model Metrics ====================
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">Model</div>
        <div class="value">{model_info.get('type', 'Random Forest')}</div>
    </div>
    """, unsafe_allow_html=True)
with col_m2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">AUC</div>
        <div class="value">{model_info.get('auc', 0.875):.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with col_m3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">Balanced Accuracy</div>
        <div class="value">{model_info.get('balanced_accuracy', 0.795):.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with col_m4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">Threshold</div>
        <div class="value">{DISPLAY_THRESHOLD:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== Mapping ====================
gender_map = {"Male": 1, "Female": 0}
race_map = {
    "Mexican American": 1,
    "Other Hispanic": 2,
    "Non-Hispanic White": 3,
    "Non-Hispanic Black": 4,
    "Other/Multiracial": 5
}
edu_map = {
    "Less than 9th grade": 1,
    "9-11th grade": 2,
    "High school diploma/GED": 3,
    "Some college/AA degree": 4,
    "College graduate or above": 5
}
pir_group_map = {
    "≤1.0": 1,
    "1.0-1.9": 2,
    "2.0-2.9": 3,
    "3.0-3.9": 4,
    "≥4.0": 5
}
smoke_map = {
    "Never": 0,
    "Former": 1,
    "Current": 2
}
activity_map = {
    "No moderate/vigorous activity": 0,
    "Has moderate/vigorous activity": 1
}
ldl_group_map = {
    "<100 mg/dL": 1,
    "≥100 mg/dL": 2
}
cancer_map = {"No": 0, "Yes": 1}
lung_map = {"No": 0, "Yes": 1}
ckm_map = {"Stage 1": 1, "Stage 2": 2, "Stage 3": 3, "Stage 4": 4}

# ==================== Main Layout ====================
col_input, col_result = st.columns([1.3, 1], gap="large")

# ==================== Input Section ====================
with col_input:
    st.markdown("### 📋 Patient Characteristics")
    st.caption("Please enter all required patient information below")

    # ===== Categorical Variables =====
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-card-title">📌 Demographics & Lifestyle</div>', unsafe_allow_html=True)

    col_gender, col_race, col_edu = st.columns(3)
    with col_gender:
        gender = st.selectbox("Gender", options=["Male", "Female"])
    with col_race:
        race = st.selectbox("Race/Ethnicity", options=list(race_map.keys()))
    with col_edu:
        edu = st.selectbox("Education Level", options=list(edu_map.keys()))

    col_pir, col_smoke, col_activity = st.columns(3)
    with col_pir:
        pir = st.selectbox("PIR Group", options=list(pir_group_map.keys()), help="Poverty Income Ratio")
    with col_smoke:
        smoke = st.selectbox("Smoking Status", options=list(smoke_map.keys()))
    with col_activity:
        activity = st.selectbox(
            "Moderate/Vigorous Physical Activity",
            options=list(activity_map.keys()),
            help="Moderate activity: brisk walking, cycling, etc. | Vigorous activity: running, swimming, etc."
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ===== Continuous Variables =====
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-card-title">📊 Clinical Measurements & Laboratory Values</div>', unsafe_allow_html=True)

    col_age, col_n, col_hgb = st.columns(3)
    with col_age:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=65, step=1)
    with col_n:
        n = st.number_input("Neutrophil Count (×10⁹/L)", min_value=0.5, max_value=20.0, value=5.0, step=0.1)
    with col_hgb:
        hgb = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=13.5, step=0.1)

    col_mcv, col_rdw, col_plt = st.columns(3)
    with col_mcv:
        mcv = st.number_input("MCV (fL)", min_value=50.0, max_value=120.0, value=90.0, step=0.5)
    with col_rdw:
        rdw = st.number_input("RDW (%)", min_value=10.0, max_value=25.0, value=13.5, step=0.1)
    with col_plt:
        plt_val = st.number_input("Platelet Count (×10⁹/L)", min_value=10, max_value=800, value=200, step=5)

    col_alb, col_glb, col_ast = st.columns(3)
    with col_alb:
        alb = st.number_input("Albumin (g/L)", min_value=15.0, max_value=55.0, value=40.0, step=0.5)
    with col_glb:
        glb = st.number_input("Globulin (g/L)", min_value=10.0, max_value=50.0, value=25.0, step=0.5)
    with col_ast:
        ast = st.number_input("AST (U/L)", min_value=1, max_value=200, value=25, step=1)

    col_acr, col_ua, col_hba1c = st.columns(3)
    with col_acr:
        acr = st.number_input("ACR (mg/g)", min_value=0.0, max_value=1000.0, value=10.0, step=1.0,
                              help="Albumin-to-Creatinine Ratio")
    with col_ua:
        ua = st.number_input("Uric Acid (mg/dL)", min_value=0.5, max_value=15.0, value=5.5, step=0.1)
    with col_hba1c:
        hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=5.7, step=0.1,
                                help="Glycated Hemoglobin")

    col_ldl, col_crp, col_lbdphs = st.columns(3)
    with col_ldl:
        ldl = st.selectbox(
            "LDL Cholesterol Level",
            options=list(ldl_group_map.keys()),
            help="Low-Density Lipoprotein Cholesterol (1 = <100 mg/dL, 2 = ≥100 mg/dL)"
        )
    with col_crp:
        crp = st.number_input("CRP (mg/L)", min_value=0.0, max_value=100.0, value=5.0, step=0.5,
                              help="C-Reactive Protein")
    with col_lbdphs:
        # ===== MODIFIED: Keep variable name LBDSPHSI, display as "LBDSPHSI (Phosphorus, mmol/L)" =====
        lbdphs = st.number_input("LBDSPHSI (Phosphorus, mmol/L)", min_value=0.0, max_value=20.0, value=5.0, step=0.1,
                                 help="Serum Phosphorus")

    col_lbxin, col_cancer, col_lung, col_ckm = st.columns(4)
    with col_lbxin:
        # ===== MODIFIED: Keep variable name LBXIN, display as "LBXIN (Serum Insulin, uU/mL)" =====
        lbxin = st.number_input("LBXIN (Serum Insulin, uU/mL)", min_value=0.0, max_value=20.0, value=1.0, step=0.1,
                                help="Serum Insulin")
    with col_cancer:
        cancer = st.selectbox("Cancer History", options=["No", "Yes"])
    with col_lung:
        lung = st.selectbox("Lung Disease", options=["No", "Yes"])
    with col_ckm:
        ckm = st.selectbox("CKM Stage", options=list(ckm_map.keys()),
                           help="Cardiovascular-Kidney-Metabolic Syndrome Stage")

    col_absi, col_sii, col_shr = st.columns(3)
    with col_absi:
        absi = st.number_input("ABSI Index", min_value=0.0, max_value=2.0, value=0.5, step=0.01,
                               help="A Body Shape Index")
    with col_sii:
        sii = st.number_input("SII (×10⁹/L)", min_value=0, max_value=5000, value=500, step=50,
                              help="Systemic Immune-Inflammation Index")
    with col_shr:
        shr = st.number_input("SHR", min_value=0, max_value=5000, value=1000, step=50,
                              help="Systemic Hemodynamic Ratio")

    st.markdown('</div>', unsafe_allow_html=True)

    predict_clicked = st.button("🔍 Predict 10-Year Mortality Risk", type="primary", use_container_width=True)


# ==================== Data Processing ====================
def create_input_data():
    """Create input DataFrame from user inputs"""
    input_dict = {
        'GENDER': gender_map[gender],
        'AGE': age,
        'RACE': race_map[race],
        'EDU': edu_map[edu],
        'PIR_GROUP': pir_group_map[pir],
        'SMOKE': smoke_map[smoke],
        'ACTIVITY': activity_map[activity],
        'N': n,
        'HGB': hgb,
        'MCV': mcv,
        'RDW': rdw,
        'PLT': plt_val,
        'ALB': alb,
        'GLB': glb,
        'AST': ast,
        'ACR': acr,
        'UA': ua,
        'HBA1C': hba1c,
        'LDL_GROUP': ldl_group_map[ldl],
        'CRP': crp,
        'LBDSPHSI': lbdphs,  # Variable name matches pkl file
        'LBXIN': lbxin,      # Variable name matches pkl file
        'CANCER': cancer_map[cancer],
        'LUNG': lung_map[lung],
        'CKM': ckm_map[ckm],
        'ABSI': absi,
        'SII': sii,
        'SHR': shr
    }
    return pd.DataFrame([input_dict])


def predict_risk(input_df):
    """Predict mortality risk probability"""
    cont_df = input_df[continuous_features]
    cat_df = input_df[categorical_features]
    cont_scaled = scaler.transform(cont_df)
    X_scaled = np.hstack([cont_scaled, cat_df.values])
    prob = model.predict_proba(X_scaled)[0, 1]
    return prob


# ==================== Results Section ====================
with col_result:
    st.markdown("### 🎯 Prediction Result")

    if predict_clicked:
        try:
            input_data = create_input_data()
            prob = predict_risk(input_data)

            # Calculate death and survival probabilities
            death_prob = prob * 100
            survival_prob = (1 - prob) * 100

            # ===== Result Card (removed risk level text) =====
            st.markdown(f"""
            <div class="result-card">
                <div class="result-number">
                    {death_prob:.1f}<span class="percent">%</span>
                </div>
                <div class="result-label">Estimated 10-Year Mortality Probability</div>
                <div class="result-prob-detail">
                    Predicted Death: {death_prob:.1f}% &nbsp;|&nbsp; Predicted Survival: {survival_prob:.1f}%
                </div>
                <div class="probability-bar">
                    <div class="death-bar" style="width: {death_prob}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: #6c757d; margin-top: 0.2rem;">
                    <span>0%</span>
                    <span>{DISPLAY_THRESHOLD * 100:.0f}% (Threshold)</span>
                    <span>100%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ===== Threshold Note =====
            st.markdown(f"""
            <div class="threshold-note">
                🔍 Classification Threshold: <strong>{DISPLAY_THRESHOLD:.2f}</strong> 
                (≥ {DISPLAY_THRESHOLD * 100:.0f}% = High Risk)
            </div>
            """, unsafe_allow_html=True)

            # ===== Interpretation =====
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📖 Clinical Interpretation")

            if prob >= DISPLAY_THRESHOLD:
                interpret_class = "interpret-death"
                interpret_title = "⚠️ High 10-Year Mortality Risk Detected"
                interpret_text = f"Probability ≥ {DISPLAY_THRESHOLD * 100:.0f}%. Consider comprehensive clinical evaluation, specialist consultation, and intensive risk factor management."
            else:
                interpret_class = "interpret-survive"
                interpret_title = "✅ Low 10-Year Mortality Risk Detected"
                interpret_text = f"Probability < {DISPLAY_THRESHOLD * 100:.0f}%. Continue routine monitoring, maintain healthy lifestyle, and adhere to standard care protocols."

            st.markdown(f"""
            <div class="interpret-box {interpret_class}">
                <p class="interpret-title" style="color: {'#721c24' if prob >= DISPLAY_THRESHOLD else '#155724'};">
                    {interpret_title}
                </p>
                <p class="interpret-text" style="color: {'#721c24' if prob >= DISPLAY_THRESHOLD else '#155724'};">
                    {interpret_text}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ===== Additional Risk Info =====
            if prob >= 0.7:
                st.warning("🚨 **Very High Risk Alert**: Probability exceeds 70%. Urgent clinical evaluation strongly recommended.")
            elif prob >= 0.5:
                st.warning("⚠️ **High Risk Alert**: Probability ≥ 50%. Clinical evaluation recommended.")
            elif prob >= 0.3:
                st.info("📊 **Moderate Risk**: Consider preventive interventions and closer monitoring.")
            else:
                st.success("✅ **Low Risk**: Continue routine preventive care.")

            # ===== Note =====
            st.markdown("""
            <div class="info-note">
                ⚠️ This tool is intended for clinical research reference only. 
                Not a substitute for professional medical diagnosis or clinical judgment.
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Please check that all inputs are valid and the model is properly loaded.")

    else:
        # ===== Placeholder =====
        st.markdown("""
        <div class="placeholder">
            <p class="placeholder-icon">🔬</p>
            <p class="placeholder-text">
                Enter all patient characteristics<br>and click <strong>"Predict 10-Year Mortality Risk"</strong>
            </p>
            <p style="font-size: 0.75rem; color: #6c757d; margin-top: 0.5rem;">
                The model uses 29 clinical features for prediction
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==================== Global Feature Importance ====================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("### 📈 Global Feature Importance Analysis")
st.caption("Feature importance based on the Random Forest model - showing the relative contribution of each clinical variable to the prediction")

try:
    importance = model.feature_importances_
    feature_names = continuous_features + categorical_features

    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors_imp = plt.cm.Blues(np.linspace(0.3, 0.9, len(imp_df)))[::-1]
    ax.barh(imp_df['Feature'], imp_df['Importance'], color=colors_imp, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest Feature Importance Ranking (10-Year Mortality Model)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, (idx, row) in enumerate(imp_df.iterrows()):
        ax.text(row['Importance'] + 0.002, i, f"{row['Importance']:.3f}",
                va='center', fontsize=8, color='#1a1a2e')

    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("📋 View Complete Feature Importance Details"):
        st.dataframe(
            imp_df.sort_values('Importance', ascending=False),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### 🔝 Top 10 Most Important Features")
        top10 = imp_df.sort_values('Importance', ascending=False).head(10)
        st.dataframe(top10, use_container_width=True, hide_index=True)

except Exception as e:
    st.warning(f"⚠️ Unable to display feature importance: {str(e)}")


# ==================== User Guide ====================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("### 📖 User Guide & Model Information")

col_help1, col_help2, col_help3 = st.columns(3)

with col_help1:
    st.markdown("""
    **📝 How to Use**
    1. Enter patient demographics (gender, race, education, etc.)
    2. Provide lifestyle information (smoking, physical activity)
    3. Input all clinical measurements and laboratory values
    4. Click **"Predict 10-Year Mortality Risk"**
    5. Review the risk probability and clinical interpretation

    ⚡ **Note**: All fields must be completed for accurate prediction.
    """)

with col_help2:
    st.markdown("""
    **📊 Understanding Results**
    - **Mortality Probability**: Estimated 10-year risk (0-100%)
    - **Risk Classification**: 
      - High Risk: ≥ 50% probability
      - Low Risk: < 50% probability
    - **Visual Bar**: Shows probability distribution
    - **Clinical Interpretation**: Tailored recommendations based on risk level

    📈 **Feature Importance**: Shows which variables most influence the prediction.
    """)

with col_help3:
    st.markdown(f"""
    **💡 Model Technical Details**
    - Algorithm: Random Forest Classifier
    - Training: {model_info.get('training_data', 'NHANES-US')} cohort
    - External Validation: {model_info.get('external_data', 'NHANES-CE')}
    - Total Features: 29 clinical variables
    - Optimal Threshold: {model_info.get('threshold', 0.1869):.4f} (external validation)
    - Display Threshold: {DISPLAY_THRESHOLD:.2f} (binary classification)

    **Performance Metrics**:
    - AUC: {model_info.get('auc', 0.875):.3f}
    - Balanced Accuracy: {model_info.get('balanced_accuracy', 0.795):.3f}
    - Cross-validation AUC: {model_info.get('cv_auc', 0.870):.3f}
    """)


# ==================== Additional Notes ====================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
with st.expander("ℹ️ Important Clinical Notes & Limitations"):
    st.markdown("""
    **Clinical Use Considerations:**
    - This prediction tool is designed for **research and educational purposes** only
    - Should not be used as the sole basis for clinical decision-making
    - Results should be interpreted by qualified healthcare professionals
    - Individual patient outcomes may vary significantly from model predictions

    **Model Limitations:**
    - Based on population-level data; individual risk may differ
    - Does not account for acute medical conditions or recent interventions
    - Missing or inaccurate input data will affect prediction accuracy
    - Assumes independent effect of each variable (no complex interactions fully captured)

    **Recommended Actions:**
    - For high-risk predictions: Comprehensive clinical evaluation, specialist referral, and risk factor modification
    - For low-risk predictions: Continue routine monitoring and preventive care
    - Always consider clinical context, patient preferences, and other relevant factors
    - Regular reassessment of risk factors is recommended

    **Version Information:**
    - Model Version: v1.0 (10-Year Mortality)
    - Last Updated: 2026
    - Cohort: NHANES 2011-2018
    """)


# ==================== Footer ====================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.caption(
    "⚠️ This tool is for clinical research reference only. Not for final diagnosis. | "
    "Model: Random Forest (10-Year) v1.0 | "
    f"Features: 29 | "
    f"Threshold: {DISPLAY_THRESHOLD:.2f} | "
    "© 2026 CKM Risk Assessment Tool"
)