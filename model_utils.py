"""
Utility functions and SalaryPredictor model wrapper for Salary Prediction ML project.
IBM Internship Project / Bharat Cares
"""

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib


def extract_seniority(title: str) -> str:
    """Extract seniority level from job title."""
    if not isinstance(title, str) or not title.strip():
        return "Mid"
    t = title.lower()
    if any(k in t for k in ["ceo", "cto", "cfo", "cdo", "vp", "director", "chief", "executive"]):
        return "Executive"
    elif any(k in t for k in ["senior", "principal", "lead", "sr.", "sr "]):
        return "Senior"
    elif any(k in t for k in ["junior", "assistant", "coordinator", "intern", "clerk", "rep", "representative", "entry", "jr.", "jr "]):
        return "Junior"
    else:
        return "Mid"


def extract_department(title: str) -> str:
    """Extract department/industry domain from job title."""
    if not isinstance(title, str) or not title.strip():
        return "Other"
    t = title.lower()
    if any(k in t for k in ["data", "scientist", "analyst", "bi", "intelligence", "machine learning", "ai"]):
        return "Data & Analytics"
    elif any(k in t for k in ["software", "developer", "engineer", "it", "web", "tech", "architect", "network", "programmer"]):
        return "Engineering & IT"
    elif any(k in t for k in ["sales", "marketing", "social media", "advertising", "public relations", "content", "copywriter", "seo"]):
        return "Sales & Marketing"
    elif any(k in t for k in ["hr", "recruiter", "human resource", "human capital", "talent", "training"]):
        return "HR & Talent"
    elif any(k in t for k in ["finance", "financial", "accountant", "accounting", "advisor", "audit"]):
        return "Finance & Accounting"
    elif any(k in t for k in ["operations", "supply chain", "project", "product", "strategy", "manager", "operations manager"]):
        return "Management & Operations"
    elif any(k in t for k in ["design", "ux", "ui", "creative", "graphic"]):
        return "Design & Creative"
    elif any(k in t for k in ["customer", "support", "help desk", "client"]):
        return "Customer Support"
    else:
        return "Management & Operations"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data and add engineered features."""
    df_feat = df.copy()
    
    # Handle missing values
    df_feat["Age"] = pd.to_numeric(df_feat.get("Age", 35.0), errors="coerce").fillna(35.0)
    df_feat["Years of Experience"] = pd.to_numeric(df_feat.get("Years of Experience", 5.0), errors="coerce").fillna(5.0)
    df_feat["Job Title"] = df_feat["Job Title"].fillna("Software Engineer").astype(str)
    df_feat["Gender"] = df_feat["Gender"].fillna("Male").astype(str)
    df_feat["Education Level"] = df_feat["Education Level"].fillna("Bachelor's").astype(str)
    
    # Standardize Education Level naming
    df_feat["Education Level"] = df_feat["Education Level"].replace({
        "Bachelor": "Bachelor's",
        "Master": "Master's",
        "phd": "PhD",
        "PHD": "PhD"
    })
    
    # Derived categorical features
    df_feat["Seniority"] = df_feat["Job Title"].apply(extract_seniority)
    df_feat["Department"] = df_feat["Job Title"].apply(extract_department)
    
    # Interaction / ratio features (safe division)
    age = df_feat["Age"].clip(lower=18)
    exp = df_feat["Years of Experience"].clip(lower=0)
    df_feat["Experience_to_Age_Ratio"] = (exp / age).round(4)
    
    return df_feat


class SalaryPredictor(BaseEstimator, RegressorMixin):
    """End-to-end Salary Prediction Estimator with automatic preprocessing."""

    def __init__(self, n_estimators=100, max_depth=12, min_samples_split=2, min_samples_leaf=1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.pipeline_ = None
        self.feature_names_in_ = None

    def _build_pipeline(self):
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

        regressor = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )

        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_eng = engineer_features(X)
        self.feature_names_in_ = list(X.columns)
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X_eng, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise ValueError("Model has not been fitted yet!")
        X_eng = engineer_features(X)
        return self.pipeline_.predict(X_eng)

    def predict_single(self, age: float, gender: str, education: str, job_title: str, experience: float) -> float:
        """Helper to predict salary for a single profile."""
        input_df = pd.DataFrame([{
            "Age": float(age),
            "Gender": str(gender),
            "Education Level": str(education),
            "Job Title": str(job_title),
            "Years of Experience": float(experience)
        }])
        pred = self.predict(input_df)
        return float(pred[0])

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        return joblib.load(filepath)
