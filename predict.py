"""
Salary Prediction Inference CLI and Module
IBM Internship Project / Bharat Cares
"""

import argparse
import os
import sys
import joblib
import pandas as pd
from model_utils import engineer_features


def load_model(model_path: str = None):
    """Load the trained model pipeline."""
    if model_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "outputs", "best_salary_model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please run train_models.py first!"
        )
    return joblib.load(model_path)


def predict_single(
    age: float,
    gender: str,
    education_level: str,
    job_title: str,
    years_of_experience: float,
    model=None
) -> dict:
    """Predict salary for a single candidate profile."""
    if model is None:
        model = load_model()

    input_df = pd.DataFrame([{
        "Age": float(age),
        "Gender": str(gender),
        "Education Level": str(education_level),
        "Job Title": str(job_title),
        "Years of Experience": float(years_of_experience)
    }])

    # Pre-calculate engineered features
    df_feat = engineer_features(input_df)

    # Predict
    predicted_salary = float(model.predict(df_feat)[0])

    # Reasonable estimation intervals (+/- 8% based on MAPE)
    lower_bound = round(predicted_salary * 0.92, 2)
    upper_bound = round(predicted_salary * 1.08, 2)

    return {
        "Age": age,
        "Gender": gender,
        "Education Level": education_level,
        "Job Title": job_title,
        "Years of Experience": years_of_experience,
        "Seniority": df_feat["Seniority"].iloc[0],
        "Department": df_feat["Department"].iloc[0],
        "Predicted Salary": round(predicted_salary, 2),
        "Salary Lower Bound (95% approx)": lower_bound,
        "Salary Upper Bound (95% approx)": upper_bound,
    }


def predict_batch(csv_path: str, output_path: str = None, model=None) -> pd.DataFrame:
    """Predict salary for multiple profiles in a CSV file."""
    if model is None:
        model = load_model()

    df = pd.read_csv(csv_path)
    df_feat = engineer_features(df)
    preds = model.predict(df_feat)
    df["Predicted_Salary"] = [round(float(p), 2) for p in preds]

    if output_path is None:
        base, ext = os.path.splitext(csv_path)
        output_path = f"{base}_with_predictions{ext}"

    df.to_csv(output_path, index=False)
    print(f"Batch predictions saved to: {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Predict Employee Salary using trained ML Model")
    parser.add_argument("--age", type=float, help="Age in years (e.g., 32)")
    parser.add_argument("--gender", type=str, choices=["Male", "Female"], help="Gender")
    parser.add_argument("--education", type=str, choices=["Bachelor's", "Master's", "PhD"], help="Education Level")
    parser.add_argument("--job", type=str, help="Job Title (e.g., 'Data Scientist', 'Software Engineer')")
    parser.add_argument("--experience", type=float, help="Years of Experience (e.g., 5.0)")
    parser.add_argument("--batch-file", type=str, help="Path to input CSV for batch prediction")
    parser.add_argument("--output-file", type=str, help="Path to output CSV for batch prediction")

    args = parser.parse_args()

    # Case 1: Batch prediction
    if args.batch_file:
        print(f"Running batch prediction on: {args.batch_file}")
        predict_batch(args.batch_file, args.output_file)
        return

    # Case 2: Direct CLI parameters
    if args.age and args.gender and args.education and args.job is not None and args.experience is not None:
        result = predict_single(args.age, args.gender, args.education, args.job, args.experience)
        print("\n" + "=" * 55)
        print("          SALARY PREDICTION RESULT")
        print("=" * 55)
        for k, v in result.items():
            if "Salary" in k:
                print(f"  {k:32s}: ${v:,.2f}")
            else:
                print(f"  {k:32s}: {v}")
        print("=" * 55 + "\n")
        return

    # Case 3: Interactive demo with pre-set sample profiles
    print("\n" + "=" * 65)
    print("      IBM BOB INTERNSHIP PROJECT - SALARY PREDICTOR DEMO")
    print("=" * 65)
    test_cases = [
        {"Age": 28, "Gender": "Female", "Education Level": "Master's", "Job Title": "Data Analyst", "Years of Experience": 3.0},
        {"Age": 35, "Gender": "Male", "Education Level": "Master's", "Job Title": "Senior Software Engineer", "Years of Experience": 9.0},
        {"Age": 45, "Gender": "Male", "Education Level": "PhD", "Job Title": "Director of Data Science", "Years of Experience": 18.0},
        {"Age": 24, "Gender": "Female", "Education Level": "Bachelor's", "Job Title": "Junior Marketing Coordinator", "Years of Experience": 1.0},
    ]

    model = load_model()
    for idx, case in enumerate(test_cases, 1):
        res = predict_single(
            case["Age"],
            case["Gender"],
            case["Education Level"],
            case["Job Title"],
            case["Years of Experience"],
            model=model
        )
        print(f"\n[Case #{idx}] {res['Job Title']} | Exp: {res['Years of Experience']} yrs | Edu: {res['Education Level']}")
        print(f"  -> Predicted Salary:  ${res['Predicted Salary']:,.2f}")
        print(f"  -> Estimated Range:   ${res['Salary Lower Bound (95% approx)']:,.2f} - ${res['Salary Upper Bound (95% approx)']:,.2f}")
        print(f"  -> Seniority Tier:    {res['Seniority']} | Domain: {res['Department']}")

    print("\n" + "=" * 65)
    print("Usage examples:")
    print("  python predict.py --age 30 --gender Male --education Master's --job \"Data Scientist\" --experience 5")
    print("  python predict.py --batch-file \"Salary Data.csv\"")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
