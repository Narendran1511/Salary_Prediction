"""
End-to-End Machine Learning Pipeline for Salary Prediction
IBM Internship Project / Bharat Cares
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import joblib

from model_utils import engineer_features, extract_department, extract_seniority


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load dataset, handle missing values and outliers/typos."""
    print("=" * 70)
    print("STEP 1: LOADING & CLEANING DATA")
    print("=" * 70)
    df = pd.read_csv(filepath)
    print(f"Initial raw shape: {df.shape}")

    # 1. Drop completely null rows
    null_rows_count = df.isnull().all(axis=1).sum()
    df_clean = df.dropna(subset=["Age", "Salary"]).copy()
    print(f"Dropped {null_rows_count} completely empty rows. Clean rows count: {len(df_clean)}")

    # 2. Fix known data entry typo: Salary = 350 for 1.5 yrs experience ($35,000)
    typo_mask = df_clean["Salary"] < 1000
    if typo_mask.any():
        typo_count = typo_mask.sum()
        df_clean.loc[typo_mask, "Salary"] = df_clean.loc[typo_mask, "Salary"] * 100
        print(f"Fixed {typo_count} salary entry typo(s) (< $1,000 scaled to standard salary range).")

    # 3. Standardize education levels
    df_clean["Education Level"] = df_clean["Education Level"].replace({
        "Bachelor": "Bachelor's",
        "Master": "Master's",
        "phd": "PhD",
        "PHD": "PhD"
    })

    print(f"Final clean dataset shape: {df_clean.shape}")
    print(f"Salary Range: ${df_clean['Salary'].min():,.0f} to ${df_clean['Salary'].max():,.0f}")
    print(f"Mean Salary:  ${df_clean['Salary'].mean():,.2f}")
    print(f"Median Salary: ${df_clean['Salary'].median():,.2f}")
    return df_clean


def generate_eda_plots(df: pd.DataFrame, output_dir: str):
    """Generate and save EDA plots."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # Plot 1: Experience vs Salary by Education
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="Years of Experience",
        y="Salary",
        hue="Education Level",
        style="Gender",
        s=80,
        alpha=0.85
    )
    plt.title("Salary vs Years of Experience by Education Level", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Years of Experience", fontsize=12)
    plt.ylabel("Salary (USD)", fontsize=12)
    plt.gca().yaxis.set_major_formatter("${x:,.0f}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_salary_vs_experience.png"), dpi=300)
    plt.close()

    # Plot 2: Salary Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df["Salary"], kde=True, color="#2b5c8f", bins=25)
    plt.title("Salary Distribution Histogram & KDE", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Salary (USD)", fontsize=12)
    plt.gca().xaxis.set_major_formatter("${x:,.0f}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_salary_distribution.png"), dpi=300)
    plt.close()
    print("Generated and saved EDA plots.")


def train_and_evaluate_all_models(df: pd.DataFrame, output_dir: str):
    """Train 9 regression models, evaluate with metrics and cross-validation, save best model."""
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING & DATASET PREPARATION")
    print("=" * 70)

    # Engineer features
    df_eng = engineer_features(df)
    features = ["Age", "Gender", "Education Level", "Years of Experience", "Experience_to_Age_Ratio", "Seniority", "Department", "Job Title"]
    X = df_eng[features]
    y = df_eng["Salary"]

    print("Engineered Features:", list(X.columns))
    print("Seniority distribution:\n", df_eng["Seniority"].value_counts())
    print("Department distribution:\n", df_eng["Department"].value_counts())

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\nTrain set size: {len(X_train)} | Test set size: {len(X_test)}")

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                ["Age", "Years of Experience", "Experience_to_Age_Ratio"]
            ),
            (
                "ord_edu",
                OrdinalEncoder(
                    categories=[["Bachelor's", "Master's", "PhD"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ),
                ["Education Level"]
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                ["Gender", "Seniority", "Department", "Job Title"]
            )
        ]
    )

    # Candidate Models
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_split=2, min_samples_leaf=1, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=120, max_depth=12, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0),
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(alpha=50.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "Support Vector Regressor": SVR(C=5000, kernel="rbf")
    }

    print("\n" + "=" * 70)
    print("STEP 3: BENCHMARKING MULTIPLE ML ALGORITHMS")
    print("=" * 70)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    fitted_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # Cross-validation on training data
        cv_r2_scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
        mean_cv_r2 = cv_r2_scores.mean()

        # Fit on training set and evaluate on test set
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        results.append({
            "Model": name,
            "CV_R2_Mean": mean_cv_r2,
            "Test_R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        })

    results_df = pd.DataFrame(results).sort_values(by="Test_R2", ascending=False).reset_index(drop=True)
    print("\n--- Model Benchmark Performance Summary ---")
    print(results_df.to_string(index=False, formatters={
        "CV_R2_Mean": "{:.4f}".format,
        "Test_R2": "{:.4f}".format,
        "MAE": "${:,.2f}".format,
        "RMSE": "${:,.2f}".format,
        "MAPE": "{:.2f}%".format
    }))

    best_model_name = results_df.iloc[0]["Model"]
    best_pipe = fitted_pipelines[best_model_name]
    best_r2 = results_df.iloc[0]["Test_R2"]
    best_mae = results_df.iloc[0]["MAE"]
    best_rmse = results_df.iloc[0]["RMSE"]
    best_mape = results_df.iloc[0]["MAPE"]

    print("\n" + "=" * 70)
    print(f"STEP 4: BEST MODEL SELECTED -> {best_model_name.upper()}")
    print("=" * 70)
    print(f"Test R² Score: {best_r2:.4f} (Explains {best_r2*100:.2f}% of salary variance)")
    print(f"Test MAE:      ${best_mae:,.2f}")
    print(f"Test RMSE:     ${best_rmse:,.2f}")
    print(f"Test MAPE:     {best_mape:.2f}%")

    # Save artifacts
    model_save_path = os.path.join(output_dir, "best_salary_model.joblib")
    joblib.dump(best_pipe, model_save_path)
    print(f"Saved best model pipeline to: {model_save_path}")

    # Save metrics JSON
    metrics_summary = {
        "best_model": best_model_name,
        "test_r2": float(best_r2),
        "test_mae": float(best_mae),
        "test_rmse": float(best_rmse),
        "test_mape": float(best_mape),
        "all_models": results_df.to_dict(orient="records")
    }
    with open(os.path.join(output_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)

    # Generate Evaluation Charts
    generate_model_charts(results_df, best_pipe, X_test, y_test, best_model_name, output_dir)

    return results_df, best_pipe


def generate_model_charts(results_df: pd.DataFrame, best_pipe, X_test: pd.DataFrame, y_test: pd.Series, best_model_name: str, output_dir: str):
    """Create comparison, actual vs predicted, and residual diagnostic charts."""
    # Chart 1: Model Comparison (R2 and MAE)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot 1: R2 Score
    sns.barplot(
        data=results_df,
        x="Test_R2",
        y="Model",
        hue="Model",
        palette="viridis",
        legend=False,
        ax=axes[0]
    )
    axes[0].set_title("Test R² Score by Model (Higher is Better)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("R² Score", fontsize=11)
    axes[0].set_xlim(0.7, 1.0)
    for p in axes[0].patches:
        val = p.get_width()
        axes[0].annotate(f"{val:.3f}", (val + 0.005, p.get_y() + p.get_height() / 2),
                         va="center", fontsize=9, fontweight="bold")

    # Subplot 2: MAE
    sns.barplot(
        data=results_df,
        x="MAE",
        y="Model",
        hue="Model",
        palette="magma",
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title("Mean Absolute Error (Lower is Better)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("MAE (USD)", fontsize=11)
    axes[1].xaxis.set_major_formatter("${x:,.0f}")
    for p in axes[1].patches:
        val = p.get_width()
        axes[1].annotate(f"${val:,.0f}", (val + 200, p.get_y() + p.get_height() / 2),
                         va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_benchmark.png"), dpi=300)
    plt.close()

    # Chart 2: Actual vs Predicted with best model
    y_pred = best_pipe.predict(X_test)
    residuals = y_test - y_pred

    plt.figure(figsize=(9, 7))
    plt.scatter(y_test, y_pred, color="#1f77b4", alpha=0.75, edgecolors="k", s=70, label="Predictions")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit (y = x)")
    plt.title(f"Actual vs Predicted Salary ({best_model_name})", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Actual Salary (USD)", fontsize=12)
    plt.ylabel("Predicted Salary (USD)", fontsize=12)
    plt.gca().xaxis.set_major_formatter("${x:,.0f}")
    plt.gca().yaxis.set_major_formatter("${x:,.0f}")
    plt.legend(frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=300)
    plt.close()

    # Chart 3: Residuals Distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(residuals, kde=True, color="#2ca02c", bins=20)
    plt.axvline(0, color="red", linestyle="--", lw=1.5, label="Zero Error")
    plt.title(f"Prediction Error (Residuals) Distribution - {best_model_name}", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Residual Error (Actual - Predicted) in USD", fontsize=12)
    plt.gca().xaxis.set_major_formatter("${x:,.0f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals_distribution.png"), dpi=300)
    plt.close()

    # Chart 4: Feature Importance if tree-based model
    if hasattr(best_pipe.named_steps["model"], "feature_importances_"):
        try:
            # Extract feature names from ColumnTransformer
            ct = best_pipe.named_steps["preprocessor"]
            feature_names = ct.get_feature_names_out()
            # Clean prefixes like 'num__', 'cat__', 'ord_edu__'
            cleaned_names = [f.split("__")[-1] for f in feature_names]
            importances = best_pipe.named_steps["model"].feature_importances_

            imp_df = pd.DataFrame({"Feature": cleaned_names, "Importance": importances})
            imp_df = imp_df.sort_values(by="Importance", ascending=False).head(15)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=imp_df, x="Importance", y="Feature", hue="Feature", palette="Blues_r", legend=False)
            plt.title(f"Top 15 Feature Importances ({best_model_name})", fontsize=14, fontweight="bold", pad=12)
            plt.xlabel("Relative Importance Score", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300)
            plt.close()
        except Exception as e:
            print("Feature importance plotting exception:", e)

    print("Saved all diagnostic and benchmark evaluation plots.")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, "Salary Data.csv")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    df_clean = load_and_clean_data(csv_file)
    generate_eda_plots(df_clean, output_dir)
    results_df, best_pipe = train_and_evaluate_all_models(df_clean, output_dir)
    print("\n" + "=" * 70)
    print("SUCCESS: ML PIPELINE TRAINING & EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
