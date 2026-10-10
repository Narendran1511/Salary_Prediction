"""
Flask Web Server & Prediction API for AI Salary Predictor Dashboard
IBM Internship Project / Bharat Cares (BOB)
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from predict import load_model, predict_single, predict_batch
from model_utils import engineer_features, extract_department, extract_seniority

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

app = Flask(__name__, static_folder=STATIC_DIR)

# Pre-load trained model and metadata
model = None
try:
    model = load_model()
    print("Loaded best_salary_model.joblib successfully.")
except Exception as e:
    print(f"Warning: Could not pre-load model: {e}")

# Pre-load analytics
analytics_data = {}
analytics_file = os.path.join(OUTPUTS_DIR, "dataset_analytics.json")
if os.path.exists(analytics_file):
    with open(analytics_file, "r") as f:
        analytics_data = json.load(f)

# Pre-load job titles
job_titles = []
titles_file = os.path.join(OUTPUTS_DIR, "job_titles.json")
if os.path.exists(titles_file):
    with open(titles_file, "r") as f:
        job_titles = json.load(f)


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict salary for single candidate profile."""
    try:
        data = request.get_json(force=True)
        age = float(data.get("age", 30))
        gender = str(data.get("gender", "Male"))
        education = str(data.get("education_level", "Bachelor's"))
        job_title = str(data.get("job_title", "Software Engineer"))
        experience = float(data.get("years_of_experience", 5))

        res = predict_single(
            age=age,
            gender=gender,
            education_level=education,
            job_title=job_title,
            years_of_experience=experience,
            model=model
        )

        # Calculate percentile relative to dataset
        predicted_salary = res["Predicted Salary"]
        percentile = 50.0
        if analytics_data and "percentiles" in analytics_data:
            p = analytics_data["percentiles"]
            p10, p25, p50, p75, p90 = p["10"], p["25"], p["50"], p["75"], p["90"]
            if predicted_salary <= p10:
                percentile = max(5.0, round((predicted_salary / p10) * 10, 1))
            elif predicted_salary <= p25:
                percentile = round(10 + (predicted_salary - p10) / (p25 - p10) * 15, 1)
            elif predicted_salary <= p50:
                percentile = round(25 + (predicted_salary - p25) / (p50 - p25) * 25, 1)
            elif predicted_salary <= p75:
                percentile = round(50 + (predicted_salary - p50) / (p75 - p50) * 25, 1)
            elif predicted_salary <= p90:
                percentile = round(75 + (predicted_salary - p75) / (p90 - p75) * 15, 1)
            else:
                percentile = min(99.0, round(90 + (predicted_salary - p90) / (250000 - p90) * 9, 1))

        # Career trajectory simulation across 0 to 20 years of experience
        trajectory = []
        for exp in [0, 1, 2, 3, 5, 7, 10, 12, 15, 18, 20]:
            sim_age = max(18.0, age - experience + exp)
            sim_res = predict_single(
                age=sim_age,
                gender=gender,
                education_level=education,
                job_title=job_title,
                years_of_experience=exp,
                model=model
            )
            trajectory.append({
                "experience": exp,
                "projected_salary": round(sim_res["Predicted Salary"], 2)
            })

        # Degrees comparison for same profile
        degree_comparison = {}
        for deg in ["Bachelor's", "Master's", "PhD"]:
            deg_res = predict_single(
                age=age,
                gender=gender,
                education_level=deg,
                job_title=job_title,
                years_of_experience=experience,
                model=model
            )
            degree_comparison[deg] = round(deg_res["Predicted Salary"], 2)

        return jsonify({
            "success": True,
            "prediction": res,
            "metrics": {
                "monthly_salary": round(predicted_salary / 12, 2),
                "hourly_rate": round(predicted_salary / (52 * 40), 2),
                "market_percentile": percentile
            },
            "trajectory": trajectory,
            "degree_comparison": degree_comparison
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/analytics", methods=["GET"])
def api_analytics():
    """Return dataset analytics, distributions, and job titles."""
    return jsonify({
        "success": True,
        "analytics": analytics_data,
        "job_titles": job_titles
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[SERVER] AI Salary Predictor Server starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
