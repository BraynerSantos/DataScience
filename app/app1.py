from flask import Flask, render_template, request
import numpy as np
import joblib
import xgboost as xgb

app = Flask(__name__)

# Load model from JSON
model = xgb.XGBRegressor()
model.load_model(r"E:\Data\CNC\Surface_Roughness\trained_model\surface_roughness_model.json")

# Load scaler
scaler = joblib.load(r"E:\Data\CNC\Surface_Roughness\trained_model\scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        try:
            # Get form inputs
            inputs = [
                float(request.form["Tool_Wear"]),
                float(request.form["Depth_of_Cut_ap"]),
                float(request.form["Cutting_Speed_vc"]),
                float(request.form["Feed_Rate_f"]),
                float(request.form["Resultant_Force"])
            ]

            # Prepare and scale input
            X_new = np.array([inputs])
            X_scaled = scaler.transform(X_new)

            # Make prediction
            y_pred = model.predict(X_scaled)
            prediction = round(float(y_pred[0]), 4)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)  