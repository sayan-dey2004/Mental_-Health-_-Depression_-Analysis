"""
MindScope - Student Depression Screening Web App
===================================================
Flask app wrapping the SVC (RBF kernel) depression-detection model
trained in model/train.py.

Configuration is environment-variable driven for production readiness:
  SECRET_KEY   - Flask session signing key (REQUIRED in production)
  FLASK_DEBUG  - "1" to enable debug mode (default: off)
  PORT         - port to listen on (default: 5000)
"""

import json
import os
import pickle
import secrets
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from flask import Flask, redirect, render_template, request, session, url_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

load_dotenv()

app = Flask(__name__)

# --- Configuration (environment-driven) -------------------------------
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config["MAX_HISTORY_ITEMS"] = int(os.environ.get("MAX_HISTORY_ITEMS", "10"))

if not os.environ.get("SECRET_KEY"):
    app.logger.warning(
        "SECRET_KEY not set in environment - using a random key generated at "
        "startup. Sessions will NOT persist across app restarts. Set SECRET_KEY "
        "as an environment variable for production deployment."
    )

CATEGORICAL_COLS = [
    "Gender",
    "City",
    "Profession",
    "Sleep Duration",
    "Dietary Habits",
    "Degree",
    "Negative Thoughts",
    "Family Stress",
]
NUMERIC_COLS = [
    "Age",
    "Academic Pressure",
    "Work Pressure",
    "CGPA",
    "Study Satisfaction",
    "Job Satisfaction",
    "Work/Study Hours",
    "Financial Stress",
]

# Direction of risk for numeric features, used only for the plain-language
# local explanation shown to the user (not used by the model itself).
# +1 -> higher value historically associated with higher depression rates
# -1 -> higher value historically associated with lower depression rates
NUMERIC_RISK_DIRECTION = {
    "Age": -1,
    "Academic Pressure": 1,
    "Work Pressure": 1,
    "CGPA": 0,  # notebook found this not significant
    "Study Satisfaction": -1,
    "Job Satisfaction": -1,
    "Work/Study Hours": 1,
    "Financial Stress": 1,
}

CATEGORICAL_RISK_LABELS = {
    "Negative Thoughts": {"Yes": "history of suicidal thoughts", "No": None},
    "Family Stress": {"Yes": "a family history of mental illness", "No": None},
    "Dietary Habits": {"Unhealthy": "unhealthy dietary habits", "Others": None, "Moderate": None, "Healthy": None},
    "Sleep Duration": {"Less than 5 hours": "insufficient sleep (under 5 hours)"},
}


# --- Load model artifacts once at startup ------------------------------
def _load_artifacts():
    with open(os.path.join(MODEL_DIR, "pipeline.pkl"), "rb") as f:
        pipeline = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(MODEL_DIR, "feature_importance.json")) as f:
        feature_importance = json.load(f)
    with open(os.path.join(MODEL_DIR, "category_options.json")) as f:
        category_options = json.load(f)
    with open(os.path.join(MODEL_DIR, "numeric_reference.json")) as f:
        numeric_reference = json.load(f)
    return pipeline, metrics, feature_importance, category_options, numeric_reference


try:
    PIPELINE, METRICS, FEATURE_IMPORTANCE, CATEGORY_OPTIONS, NUMERIC_REFERENCE = _load_artifacts()
    MODEL_LOAD_ERROR = None
except FileNotFoundError as e:
    PIPELINE = METRICS = FEATURE_IMPORTANCE = CATEGORY_OPTIONS = NUMERIC_REFERENCE = None
    MODEL_LOAD_ERROR = str(e)


NUMERIC_FIELD_SPECS = {
    "Age": {"min": 15, "max": 60, "step": 1, "label": "Age"},
    "Academic Pressure": {"min": 0, "max": 5, "step": 1, "label": "Academic Pressure (0 = none, 5 = extreme)"},
    "Work Pressure": {"min": 0, "max": 5, "step": 1, "label": "Work Pressure (0 = none, 5 = extreme)"},
    "CGPA": {"min": 0, "max": 10, "step": 0.01, "label": "CGPA (0-10 scale)"},
    "Study Satisfaction": {"min": 0, "max": 5, "step": 1, "label": "Study Satisfaction (0 = very low, 5 = very high)"},
    "Job Satisfaction": {"min": 0, "max": 5, "step": 1, "label": "Job Satisfaction (0 = very low, 5 = very high)"},
    "Work/Study Hours": {"min": 0, "max": 16, "step": 1, "label": "Work/Study Hours per day"},
    "Financial Stress": {"min": 1, "max": 5, "step": 1, "label": "Financial Stress (1 = low, 5 = extreme)"},
}


def build_explanation(input_dict, probability):
    """Rule-based local explanation: which of the user's answers line up
    with the globally most important, highest-risk-direction features.
    This is descriptive (pattern-matching against training data trends),
    not a causal or clinical claim.
    """
    top_features = [f["feature"] for f in FEATURE_IMPORTANCE if f["importance"] > 0][:6]
    contributing = []

    for feat in top_features:
        if feat in NUMERIC_COLS:
            ref = NUMERIC_REFERENCE.get(feat)
            direction = NUMERIC_RISK_DIRECTION.get(feat, 0)
            if not ref or direction == 0:
                continue
            value = input_dict.get(feat)
            if value is None:
                continue
            value = float(value)
            mean = ref["mean"]
            std = ref["std"] or 1
            z = (value - mean) / std
            if direction == 1 and z > 0.5:
                contributing.append(f"Your {feat.lower()} ({value:g}) is notably higher than the average student ({mean:g}).")
            elif direction == -1 and z < -0.5:
                contributing.append(f"Your {feat.lower()} ({value:g}) is notably lower than the average student ({mean:g}).")
        elif feat in CATEGORICAL_RISK_LABELS:
            value = input_dict.get(feat)
            label = CATEGORICAL_RISK_LABELS[feat].get(value)
            if label:
                contributing.append(f"You reported {label}.")

    return contributing


@app.route("/")
def index():
    if MODEL_LOAD_ERROR:
        return render_template("error.html", error=MODEL_LOAD_ERROR), 500
    return render_template(
        "index.html",
        category_options=CATEGORY_OPTIONS,
        numeric_specs=NUMERIC_FIELD_SPECS,
        categorical_cols=CATEGORICAL_COLS,
        numeric_cols=NUMERIC_COLS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL_LOAD_ERROR:
        return render_template("error.html", error=MODEL_LOAD_ERROR), 500

    form = request.form
    input_dict = {}

    try:
        for col in CATEGORICAL_COLS:
            val = form.get(col, "").strip()
            if not val:
                raise ValueError(f"Missing value for {col}")
            input_dict[col] = val

        for col in NUMERIC_COLS:
            raw = form.get(col, "").strip()
            if not raw:
                raise ValueError(f"Missing value for {col}")
            input_dict[col] = float(raw)
    except ValueError as e:
        return render_template(
            "index.html",
            category_options=CATEGORY_OPTIONS,
            numeric_specs=NUMERIC_FIELD_SPECS,
            categorical_cols=CATEGORICAL_COLS,
            numeric_cols=NUMERIC_COLS,
            error=str(e),
            form_values=form,
        ), 400

    X_new = pd.DataFrame([input_dict])[CATEGORICAL_COLS + NUMERIC_COLS]

    prediction = int(PIPELINE.predict(X_new)[0])
    probability = float(PIPELINE.predict_proba(X_new)[0][1])

    if probability >= 0.7:
        risk_level = "High"
    elif probability >= 0.4:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    contributing_factors = build_explanation(input_dict, probability)

    result = {
        "prediction": prediction,
        "probability": round(probability * 100, 1),
        "risk_level": risk_level,
        "contributing_factors": contributing_factors,
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "inputs": input_dict,
    }

    history = session.get("history", [])
    history.insert(0, result)
    session["history"] = history[: app.config["MAX_HISTORY_ITEMS"]]
    session.modified = True

    show_crisis_resources = input_dict.get("Negative Thoughts") == "Yes" or risk_level == "High"

    return render_template("result.html", result=result, show_crisis_resources=show_crisis_resources)


@app.route("/history")
def history():
    hist = session.get("history", [])
    return render_template("history.html", history=hist)


@app.route("/history/clear", methods=["POST"])
def clear_history():
    session.pop("history", None)
    return redirect(url_for("history"))


@app.route("/about")
def about():
    if MODEL_LOAD_ERROR:
        return render_template("error.html", error=MODEL_LOAD_ERROR), 500
    return render_template("about.html", metrics=METRICS, feature_importance=FEATURE_IMPORTANCE)


@app.route("/resources")
def resources():
    return render_template("resources.html")


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
