from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model & label encoder
model = joblib.load("employee_performance_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

@app.route("/result", methods=["POST"])
def result():

    # -------- Collect input data (NO productivity_score from user) --------
    data = {
        "task_delay_days": int(request.form["task_delay_days"]),
        "avg_task_completion_time": float(request.form["avg_task_completion_time"]),
        "department_rating": int(request.form["department_rating"]),
        "attendance_percentage": int(request.form["attendance_percentage"]),
        "days_absent": int(request.form["days_absent"]),
        "skill_level": int(request.form["skill_level"]),
        "years_experience": int(request.form["years_experience"]),
        "projects_completed": int(request.form["projects_completed"]),
        "client_feedback_score": int(request.form["client_feedback_score"]),
        "training_hours": int(request.form["training_hours"]),
        "overtime_hours": int(request.form["overtime_hours"]),
        "errors_reported": int(request.form["errors_reported"]),
        "teamwork_score": int(request.form["teamwork_score"])
    }

    # -------- Compute productivity score (SAME FORMULA AS DATASET) --------
    productivity_score = (
        data["skill_level"] * 10 +
        data["department_rating"] * 8 +
        data["client_feedback_score"] * 7 +
        data["attendance_percentage"] * 0.3 +
        data["projects_completed"] * 2 -
        data["task_delay_days"] * 2 -
        data["errors_reported"] * 3
    )

    data["productivity_score"] = round(productivity_score, 2)

    # -------- Create DataFrame in SAME FEATURE ORDER --------
    feature_order = [
        "task_delay_days",
        "avg_task_completion_time",
        "department_rating",
        "attendance_percentage",
        "days_absent",
        "skill_level",
        "years_experience",
        "projects_completed",
        "client_feedback_score",
        "training_hours",
        "overtime_hours",
        "errors_reported",
        "teamwork_score",
        "productivity_score"
    ]

    input_df = pd.DataFrame([[data[col] for col in feature_order]], columns=feature_order)

    # -------- Prediction (NO SCALING FOR RANDOM FOREST) --------
    prediction_encoded = model.predict(input_df)
    prediction = label_encoder.inverse_transform(prediction_encoded)[0]

    return render_template(
        "result.html",
        prediction=prediction,
        productivity_score=data["productivity_score"]
    )

if __name__ == "__main__":
    app.run(debug=True)
