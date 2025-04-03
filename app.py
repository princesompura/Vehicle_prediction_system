from flask import Flask, request, jsonify

import pickle
import pandas as pd
import os

app = Flask(__name__)

# Define model path inside "models" folder
model_path = os.path.join("models", "random_forest.pkl")

# Load trained model
model = pickle.load(open(model_path, "rb"))

# Define expected input features
features = ['year', 'mileage', 'cylinders', 'fuel', 'transmission', 'body', 'drivetrain']

@app.route('/')
def home():
    return """
    <h1> Vehicle Price Prediction API</h1>
    <p> The API is running! Send a POST request to <code>/predict</code> with vehicle details to get a price prediction.</p>

    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate input
        for feature in features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        # Convert input data into a DataFrame
        input_data = pd.DataFrame([data])

        input_data['fuel'] = input_data['fuel'].astype(str).str.strip().str.lower()
        input_data['transmission'] = input_data['transmission'].astype(str).str.strip().str.lower()
        input_data['body'] = input_data['body'].astype(str).str.strip().str.lower()
        input_data['drivetrain'] = input_data['drivetrain'].astype(str).str.strip().str.lower()


        # Make prediction
        prediction = model.predict(input_data)[0]

        return jsonify({"predicted_price": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Vehicle Price Prediction API is starting...")
    print("🔗 Access it at: http://127.0.0.1:5000/")
    print("📩 Send a POST request to /predict with vehicle details to get a price prediction.")
    app.run(debug=True)
