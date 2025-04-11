from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the fine-tuned Random Forest model
model_path = os.path.join("models", "random_forest.pkl")
model = pickle.load(open(model_path, "rb"))

# Define the expected input features
features = ['year', 'mileage', 'cylinders', 'fuel', 'transmission', 'body', 'drivetrain']

@app.route('/')
def home():
    return """
    <h1>Vehicle Price Prediction API</h1>
    <p>The fine-tuned Random Forest model is live!</p>
    <p>Send a POST request to <code>/predict</code> with vehicle details in JSON format.</p>
    <p>Example format:</p>
    <pre>
    {
        "year": 2024,
        "mileage": 5000,
        "cylinders": 4,
        "fuel": "Gasoline",
        "transmission": "Automatic",
        "body": "SUV",
        "drivetrain": "Four-wheel Drive"
    }
    </pre>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate input
        for feature in features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        # Convert input to DataFrame
        df_input = pd.DataFrame([data])

        # Normalize text columns
        df_input['fuel'] = df_input['fuel'].astype(str).str.strip().str.lower()
        df_input['transmission'] = df_input['transmission'].astype(str).str.strip().str.lower()
        df_input['body'] = df_input['body'].astype(str).str.strip().str.lower()
        df_input['drivetrain'] = df_input['drivetrain'].astype(str).str.strip().str.lower()

        # Make prediction
        predicted_price = model.predict(df_input)[0]

        return jsonify({'predicted_price': round(predicted_price, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Fine-tuned Vehicle Price Prediction API is starting...")
    print("🔗 Access it at: http://127.0.0.1:5000/")
    print("📩 POST your vehicle data to /predict to get a price prediction.")
    app.run(debug=True)
