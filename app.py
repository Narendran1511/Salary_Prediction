"""
Interactive Salary Prediction Web Application
IBM Internship Project / Bharat Cares (BOB)
Built with Streamlit and Scikit-Learn
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from model_utils import engineer_features, extract_department, extract_seniority
from predict import predict_single, predict_batch

# Page configuration
st.set_page_config(
    page_title="AI Salary Predictor | IBM BOB Project",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f62fe 0%, #001d6c 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(15, 98, 254, 0.2);
    }
    .metric-card {
        background: #f4f7fb;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #0f62fe;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #198038 0%, #0e6027 100%);
        color: white;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 6px 20px rgba(25, 128, 56, 0.25);
    }
    .sub-title {
        color: #525252;
        font-size: 1.1rem;
    }
    .stButton>button {
        background-color: #0f62fe;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0353e9;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get paths
@st.cache_resource
def load_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "outputs", "best_salary_model.joblib")
    metrics_path = os.path.join(base_dir, "outputs", "model_metrics.json")
    csv_path = os.path.join(base_dir, "Salary Data.csv")

    model = joblib.load(model_path) if os.path.exists(model_path) else None
    
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
    return model, metrics, df

model, metrics, df = load_resources()

# Header Banner
st.markdown("""
<div class="main-header">
    <h1 style='margin:0; font-size: 2.5rem; font-weight: 700;'>💼 AI Employee Salary Predictor</h1>
    <p style='margin-top: 0.5rem; font-size: 1.1rem; opacity: 0.9;'>
        Machine Learning System for Compensation Forecasting | IBM BOB Internship Project
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Interactive Prediction",
    "📊 Dataset Exploratory Analysis",
    "🏆 Model Performance & Benchmark",
    "📁 Batch Prediction & CSV"
])

# ----------------- TAB 1: INTERACTIVE PREDICTION -----------------
with tab1:
    st.markdown("### 🔍 Predict Compensation for a Professional Profile")
    st.write("Enter employee credentials below to compute market salary estimation powered by the tuned Random Forest Regressor ($R^2 = 0.935$).")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Candidate Information")
        age = st.slider("Employee Age (years)", min_value=18, max_value=65, value=30, step=1)
        experience = st.slider("Years of Professional Experience", min_value=0.0, max_value=40.0, value=5.0, step=0.5)
        
        # Guardrail warning
        if experience > (age - 16):
            st.warning("⚠️ Note: Years of experience appears high relative to age. Prediction will adjust accordingly.")

        gender = st.radio("Gender", options=["Male", "Female"], horizontal=True)

    with col2:
        st.subheader("Role & Qualification")
        education = st.selectbox(
            "Highest Education Level",
            options=["Bachelor's", "Master's", "PhD"],
            index=0
        )

        # Default job list or free input
        default_jobs = [
            "Software Engineer",
            "Data Scientist",
            "Data Analyst",
            "Product Manager",
            "Financial Analyst",
            "Marketing Manager",
            "HR Manager",
            "Operations Manager",
            "Director of Marketing",
            "Director of Operations",
            "Senior Software Engineer",
            "Senior Data Scientist",
            "Chief Technology Officer",
            "Business Analyst",
            "UX Designer",
            "Sales Manager",
            "Other (Custom)"
        ]
        
        selected_job = st.selectbox("Job Title / Role", options=default_jobs, index=0)
        if selected_job == "Other (Custom)":
            job_title = st.text_input("Enter Custom Job Title", value="Machine Learning Engineer")
        else:
            job_title = selected_job

        # Preview inferred metadata
        derived_sen = extract_seniority(job_title)
        derived_dept = extract_department(job_title)
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>Inferred Hierarchy:</strong> <code>{derived_sen}</code><br>
            <strong>Inferred Domain:</strong> <code>{derived_dept}</code>
        </div>
        """, unsafe_allow_html=True)

    predict_btn = st.button("🚀 Calculate Predicted Salary", use_container_width=True)

    if predict_btn or True:  # Real-time or clicked
        res = predict_single(
            age=age,
            gender=gender,
            education_level=education,
            job_title=job_title,
            years_of_experience=experience,
            model=model
        )

        pred_salary = res["Predicted Salary"]
        min_salary = res["Salary Lower Bound (95% approx)"]
        max_salary = res["Salary Upper Bound (95% approx)"]

        st.markdown(f"""
        <div class="prediction-card">
            <h3 style='margin:0; font-size:1.4rem; opacity:0.9;'>ESTIMATED MARKET SALARY</h3>
            <h1 style='margin:0.5rem 0; font-size:3.5rem; font-weight:800;'>${pred_salary:,.2f}</h1>
            <p style='margin:0; font-size:1.1rem; opacity:0.95;'>
                Estimated Band: <strong>${min_salary:,.2f}</strong> — <strong>${max_salary:,.2f}</strong> (±8% Margin)
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Insights metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Seniority Tier", res["Seniority"])
        with m_col2:
            st.metric("Domain Sector", res["Department"])
        with m_col3:
            monthly_est = pred_salary / 12
            st.metric("Est. Monthly Salary", f"${monthly_est:,.2f}")
        with m_col4:
            st.metric("Model Confidence (R²)", "93.5%")

# ----------------- TAB 2: EXPLORATORY DATA ANALYSIS -----------------
with tab2:
    st.subheader("📊 Exploratory Data Analysis & Insights")
    st.write("Exploration of patterns in the 373 professional salary records.")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    eda_exp_img = os.path.join(base_dir, "outputs", "eda_salary_vs_experience.png")
    eda_dist_img = os.path.join(base_dir, "outputs", "eda_salary_distribution.png")

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(eda_exp_img):
            st.image(eda_exp_img, caption="Salary Growth vs Years of Experience", use_container_width=True)
    with c2:
        if os.path.exists(eda_dist_img):
            st.image(eda_dist_img, caption="Overall Salary Distribution in Dataset", use_container_width=True)

    if df is not None:
        st.markdown("#### Key Dataset Statistics")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        stat_col1.metric("Total Records", f"{len(df):,}")
        stat_col2.metric("Average Salary", f"${df['Salary'].mean():,.2f}")
        stat_col3.metric("Median Salary", f"${df['Salary'].median():,.2f}")
        stat_col4.metric("Max Recorded Salary", f"${df['Salary'].max():,.2f}")

        with st.expander("👀 View Raw Sample Records"):
            st.dataframe(df.head(20), use_container_width=True)

# ----------------- TAB 3: MODEL PERFORMANCE & BENCHMARK -----------------
with tab3:
    st.subheader("🏆 Model Benchmarking & Performance Diagnostics")
    st.write("Comparison across 9 distinct machine learning regression models evaluated on cross-validation and holdout test set.")

    if metrics is not None and "all_models" in metrics:
        models_df = pd.DataFrame(metrics["all_models"])
        
        col_bench1, col_bench2 = st.columns([1.5, 1])
        with col_bench1:
            st.markdown("##### 📈 Comparison Leaderboard")
            # Format dataframe
            display_df = models_df.copy()
            display_df["CV_R2_Mean"] = display_df["CV_R2_Mean"].apply(lambda x: f"{x:.4f}")
            display_df["Test_R2"] = display_df["Test_R2"].apply(lambda x: f"{x:.4f}")
            display_df["MAE"] = display_df["MAE"].apply(lambda x: f"${x:,.2f}")
            display_df["RMSE"] = display_df["RMSE"].apply(lambda x: f"${x:,.2f}")
            display_df["MAPE"] = display_df["MAPE"].apply(lambda x: f"{x:.2f}%")
            st.dataframe(display_df, use_container_width=True)

        with col_bench2:
            st.markdown("##### 🌟 Winning Model Highlights")
            st.success(f"""
            **Selected Champion:** **{metrics.get('best_model', 'Random Forest')}**  
            - **Test R² Score:** `{metrics.get('test_r2', 0.935):.4f}`  
            - **Mean Absolute Error (MAE):** `${metrics.get('test_mae', 8144):,.2f}`  
            - **Root Mean Squared Error (RMSE):** `${metrics.get('test_rmse', 12485):,.2f}`  
            - **Mean Absolute Percentage Error (MAPE):** `{metrics.get('test_mape', 8.59):.2f}%`  
            """)

    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        bm_img = os.path.join(base_dir, "outputs", "model_comparison_benchmark.png")
        if os.path.exists(bm_img):
            st.image(bm_img, caption="Benchmark Comparison: R² and MAE", use_container_width=True)

    with c_chart2:
        avp_img = os.path.join(base_dir, "outputs", "actual_vs_predicted.png")
        if os.path.exists(avp_img):
            st.image(avp_img, caption="Actual vs Predicted Salary Scatter", use_container_width=True)

    c_chart3, c_chart4 = st.columns(2)
    with c_chart3:
        res_img = os.path.join(base_dir, "outputs", "residuals_distribution.png")
        if os.path.exists(res_img):
            st.image(res_img, caption="Residual Error Distribution", use_container_width=True)

    with c_chart4:
        fi_img = os.path.join(base_dir, "outputs", "feature_importance.png")
        if os.path.exists(fi_img):
            st.image(fi_img, caption="Random Forest Feature Importances", use_container_width=True)

# ----------------- TAB 4: BATCH PREDICTION & CSV -----------------
with tab4:
    st.subheader("📁 Batch File Prediction & Export")
    st.write("Upload a CSV file containing columns `Age`, `Gender`, `Education Level`, `Job Title`, `Years of Experience` to generate predictions in bulk.")

    uploaded_file = st.file_uploader("Upload CSV File for Prediction", type=["csv"])

    if uploaded_file is not None:
        try:
            input_csv_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded `{len(input_csv_df)}` rows. Generating predictions...")

            feat_df = engineer_features(input_csv_df)
            preds = model.predict(feat_df)
            
            output_df = input_csv_df.copy()
            output_df["Predicted_Salary"] = [round(float(p), 2) for p in preds]
            output_df["Predicted_Salary_Min"] = [round(float(p) * 0.92, 2) for p in preds]
            output_df["Predicted_Salary_Max"] = [round(float(p) * 1.08, 2) for p in preds]

            st.dataframe(output_df.head(25), use_container_width=True)

            csv_download = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_download,
                file_name="salary_predictions_result.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
    else:
        st.info("💡 You can also download the precomputed predictions on the original dataset:")
        pred_original_path = os.path.join(base_dir, "Salary Data_with_predictions.csv")
        if os.path.exists(pred_original_path):
            with open(pred_original_path, "rb") as f:
                st.download_button(
                    label="📥 Download Dataset with Predicted Salaries (Salary Data_with_predictions.csv)",
                    data=f.read(),
                    file_name="Salary_Data_with_predictions.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #777; font-size: 0.9rem;'>
    IBM Internship Project / Bharat Cares (Bank of Baroda AI Initiative) • Model: Random Forest Regressor • R²: 0.9350
</div>
""", unsafe_allow_html=True)
