📋 Project Overview
This is a web-based application for predicting 10-year all-cause mortality risk in patients with Cardiovascular-Kidney-Metabolic (CKM) syndrome. The application is built using the NHANES (National Health and Nutrition Examination Survey) database and employs a machine learning model (Random Forest) to provide personalized risk assessments for long-term mortality prediction.

🎯 Key Features
Risk Prediction: Predicts 10-year mortality probability based on 29 clinical features

Interactive Interface: User-friendly web interface built with Streamlit

Model Interpretability: Global feature importance visualization and SHAP (SHapley Additive exPlanations) value analysis to explain prediction results

Real-time Feedback: Instant risk assessment with visual indicators and probability bars

Comprehensive Feature Set: Includes demographic, lifestyle, and laboratory variables

Risk Stratification: Classifies patients into High Risk (≥50%) or Low Risk (<50%) categories

Clinical Interpretation: Provides tailored recommendations based on risk level

📊 Input Features
Categorical Variables (10 features)
Gender (Male/Female)

Race/Ethnicity (Mexican American, Other Hispanic, Non-Hispanic White, Non-Hispanic Black, Other/Multiracial)

Education Level (Less than 9th grade, 9-11th grade, High school diploma/GED, Some college/AA degree, College graduate or above)

PIR Group (<1.0, 1.0-2.9, ≥3.0)

Smoking Status (Never/Former/Current)

Moderate/Vigorous Physical Activity (No/Yes)

LDL Cholesterol Level (<100 mg/dL, ≥100 mg/dL)

Cancer History (No/Yes)

Lung Disease (No/Yes)

CKM Stage (Stage 1-4)

Continuous Variables (19 features)
Age (years)

Neutrophil Count (N, ×10⁹/L)

Hemoglobin (HGB, g/dL)

MCV (Mean Corpuscular Volume, fL)

RDW (Red Cell Distribution Width, %)

PLT (Platelet Count, ×10⁹/L)

Albumin (ALB, g/L)

Globulin (GLB, g/L)

AST (Aspartate Aminotransferase, U/L)

ACR (Albumin-to-Creatinine Ratio, mg/g)

Uric Acid (UA, mg/dL)

HbA1c (Glycated Hemoglobin, %)

CRP (C-Reactive Protein, mg/L)

LBDSPHSI (Serum Phosphorus, mmol/L)

LBXIN (Serum Insulin, uU/L))

ABSI (A Body Shape Index)

SII (Systemic Immune-Inflammation Index, ×10⁹/L)

SHR (Systemic Hemodynamic Ratio)

🚀 Model Performance
Algorithm: Random Forest Classifier

External Validation AUC: 0.875
External Balanced Accuracy: 79.5%
Features: 29 clinical parameters

External Balanced Accuracy: 79.5%
