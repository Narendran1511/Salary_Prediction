# 💼 IBM BOB Internship Project: Employee Salary Prediction using Machine Learning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.9.0-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.60.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Status](https://img.shields.io/badge/Status-Completed-success)](#)

> **Initiative:** IBM SkillsBuild & Bharat Cares (Bank of Baroda AI Track)  
> **Topic:** Machine Learning Compensation Forecasting & Workforce Analytics  
> **Primary Target:** Predicting `Salary` based on demographic and professional attributes.

---

## 📌 Executive Summary

Determining competitive and equitable employee compensation is essential for organizational retention and talent acquisition. This project implements an end-to-end Machine Learning pipeline to predict employee salaries from historical market data.

Our solution includes:
1. **Automated Data Cleaning & Quality Control:** Null row elimination, typographical anomaly correction, and categorical standardization.
2. **Domain-Specific Feature Engineering:** Natural language heuristic extraction of **Seniority** and **Department** tiers, and experience-to-age interaction metrics.
3. **Multi-Model Benchmarking:** 9 regression algorithms trained, tuned, and evaluated with 5-fold cross-validation and holdout validation.
4. **Winning Model Selection:** Tuned **Random Forest Regressor** achieving an **$R^2$ Score of 0.9350** (explaining 93.5% of variance) with a low **Mean Absolute Error (MAE) of $8,144.68** (~8.59% MAPE).
5. **Interactive Streamlit Web Dashboard (`app.py`):** Real-time salary calculation, interactive EDA charts, leaderboard diagnostics, and CSV batch processing.
6. **Command-Line & Batch Inference CLI (`predict.py`):** Single-command CLI inference and automated CSV batch prediction.
7. **Jupyter Notebook Submission (`Salary_Prediction_Project.ipynb`):** Step-by-step reproducible notebook for review.

---

## 📁 Repository Structure

```text
IBM_BOB_Project/
├── Salary Data.csv                     # Original dataset (375 records)
├── Salary Data_with_predictions.csv    # Original dataset with ML predictions appended
├── train_models.py                     # Complete training & model evaluation script
├── model_utils.py                      # Feature engineering & estimator utilities
├── predict.py                          # Prediction inference CLI (single & batch)
├── app.py                              # Modern interactive Streamlit web dashboard
├── Salary_Prediction_Project.ipynb     # Comprehensive step-by-step Jupyter Notebook
├── README.md                           # Complete project documentation
└── outputs/                            # Artifacts & visualization directory
    ├── best_salary_model.joblib        # Serialized winning model pipeline
    ├── model_metrics.json              # Evaluation metrics for all 9 algorithms
    ├── eda_salary_vs_experience.png    # Scatter plot: Experience vs Salary
    ├── eda_salary_distribution.png     # Histogram: Salary distribution
    ├── model_comparison_benchmark.png  # Comparison bar charts: R² & MAE
    ├── actual_vs_predicted.png         # Regression scatter plot (y = x fit)
    ├── residuals_distribution.png      # Error residual distribution
    └── feature_importance.png          # Top 15 Random Forest feature importances
```

---

## 🔬 Dataset & Data Cleaning

### Initial Dataset Overview
- **Raw Entries:** 375 rows, 6 columns
- **Features:** `Age`, `Gender`, `Education Level`, `Job Title`, `Years of Experience`
- **Target Variable:** `Salary` (continuous in USD)

### Key Cleaning Steps
1. **Empty Records Handling:** 2 rows (indices 172 and 260) contained all `NaN` values and were dropped.
2. **Anomaly & Typo Correction:** Row 259 (Salary = 350 for 1.5 yrs experience as Junior Business Operations Analyst) was an obvious data entry typo where two zeros were omitted compared to duplicate record at row 220 ($35,000). Scaled entry to $35,000.
3. **Categorical Standardization:** Standardized education level values (`Bachelor` -> `Bachelor's`).
4. **Clean Dataset Summary:**
   - Total clean records: **373**
   - Minimum salary: **$30,000.00**
   - Median salary: **$95,000.00**
   - Average salary: **$100,670.24**
   - Maximum salary: **$250,000.00**

---

## 🛠 Feature Engineering Pipeline

Because `Job Title` contains 174 unique titles across 373 rows, standard one-hot encoding alone risks high-cardinality overfitting. We engineered hierarchical and domain features:

| Feature | Type | Description / Logic |
| :--- | :--- | :--- |
| `Seniority` | Categorical | Extracted from titles: `Executive` (CEO, VP, Director), `Senior` (Lead, Principal), `Mid` (Analyst, Engineer), `Junior` (Entry, Assistant, Coordinator) |
| `Department` | Categorical | Extracted domain: `Data & Analytics`, `Engineering & IT`, `Sales & Marketing`, `Management & Operations`, `HR & Talent`, `Finance & Accounting`, `Design & Creative` |
| `Experience_to_Age_Ratio` | Numerical | `Years of Experience / Age` (indicates career progression pace) |
| `Education Level` | Ordinal | Encoded monotonically: `Bachelor's (0) < Master's (1) < PhD (2)` |

---

## 🏆 Model Benchmarking & Performance Leaderboard

All models were evaluated on the exact same 80/20 train/test split (`random_state=42`) and validated using 5-fold cross-validation.

| Rank | Model | 5-Fold CV $R^2$ | Test $R^2$ | Test MAE ($) | Test RMSE ($) | Test MAPE (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Random Forest Regressor** | **0.9340** | **0.9350** | **$8,144.68** | **$12,485.79** | **8.59%** |
| 🥈 | **Extra Trees Regressor** | 0.9352 | 0.9275 | $8,009.33 | $13,187.10 | 8.60% |
| 🥉 | **Ridge Regression** | 0.9447 | 0.9258 | $8,992.06 | $13,334.98 | 9.85% |
| 4 | **Linear Regression** | 0.9388 | 0.9255 | $8,991.31 | $13,368.14 | 9.98% |
| 5 | **Lasso Regression** | 0.9450 | 0.9208 | $9,284.83 | $13,775.72 | 10.27% |
| 6 | **K-Nearest Neighbors** | 0.9328 | 0.9195 | $9,031.67 | $13,888.36 | 10.49% |
| 7 | **Gradient Boosting** | 0.9342 | 0.9154 | $8,926.45 | $14,245.40 | 9.33% |
| 8 | **Support Vector Regressor** | 0.9221 | 0.8930 | $10,033.42 | $16,020.64 | 10.73% |
| 9 | **Decision Tree** | 0.9076 | 0.8846 | $10,618.31 | $16,631.50 | 11.01% |

### Key Findings
1. **Random Forest** achieved the best overall generalization with an **$R^2$ of 0.9350**, capturing non-linear seniority ceilings and salary jumps.
2. Linear models (Ridge, OLS) also showed strong linear performance ($R^2 \approx 0.925$) due to the high fundamental correlation between Years of Experience and Salary ($r = 0.931$).
3. The model exhibits a tight residual spread centered symmetrically around $0 without skew.

---

## 🚀 How to Run the Project

### 1. Run Model Training & Evaluation
Re-trains all 9 models, evaluates cross-validation, produces plots, and exports `best_salary_model.joblib`:
```powershell
python train_models.py
```

### 2. Predict Salary for a Custom Profile (CLI)
```powershell
python predict.py --age 32 --gender Male --education "Master's" --job "Senior Software Engineer" --experience 8
```
Sample Output:
```text
=======================================================
          SALARY PREDICTION RESULT
=======================================================
  Age                             : 32.0
  Gender                          : Male
  Education Level                 : Master's
  Job Title                       : Senior Software Engineer
  Years of Experience             : 8.0
  Seniority                       : Senior
  Department                      : Engineering & IT
  Predicted Salary                : $103,450.00
  Salary Lower Bound (95% approx) : $95,174.00
  Salary Upper Bound (95% approx) : $111,726.00
=======================================================
```

### 3. Run Batch Predictions on a CSV File
```powershell
python predict.py --batch-file "Salary Data.csv"
```
Generates `Salary Data_with_predictions.csv` with predictions attached to all rows.

### 4. Launch Interactive Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501` to use the interactive prediction sliders, dataset visualizer, model leaderboard, and batch CSV uploader.

### 5. Open Jupyter Notebook
```powershell
jupyter notebook Salary_Prediction_Project.ipynb
```

---

## 📊 Summary of Model Artifacts in `outputs/`
- `best_salary_model.joblib`: Tuned production pipeline ready for deployment
- `model_metrics.json`: Exported performance JSON with all cross-validation and test scores
- `model_comparison_benchmark.png`: Bar chart comparison of all algorithms
- `actual_vs_predicted.png`: Scatter plot comparing predicted salary against true ground truth
- `residuals_distribution.png`: Error distribution confirming normality
- `feature_importance.png`: Importance breakdown showing dominant predictive features

---
**IBM Internship Project & Bharat Cares • 2026**
