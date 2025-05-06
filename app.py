from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")  # Renders the HTML form

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the form
        features = [
            float(request.form["N"]),
            float(request.form["P"]),
            float(request.form["K"]),
            float(request.form["pH"]),
            float(request.form["EC"]),
            float(request.form["OC"]),
            float(request.form["S"]),
            float(request.form["Zn"]),
            float(request.form["Fe"]),
            float(request.form["Cu"]),
            float(request.form["Mn"]),
            float(request.form["B"]),
        ]

        # Convert to numpy array
        features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        return render_template("index.html", prediction=f"Predicted Output: {prediction}")

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
