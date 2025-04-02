from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('../models/random_forest.pkl', 'rb'))

# Define feature columns
num_features = ['year', 'mileage', 'cylinders']
cat_features = ['fuel', 'transmission', 'body', 'drivetrain']

@app.route('/')
def home():
    return jsonify({'message': 'Vehicle Price Prediction API is Running!'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json

        # Convert JSON into DataFrame
        input_data = pd.DataFrame([data])

        # Load the original dataset for encoding reference
        df = pd.read_csv('../dataset/dataset.csv')

        # Fill missing values
        for col in num_features:
            df[col].fillna(df[col].median(), inplace=True)
        for col in cat_features:
            df[col].fillna(df[col].mode()[0], inplace=True)

        # Concatenate the input data with the dataset for encoding consistency
        df = pd.concat([df[cat_features], input_data[cat_features]], ignore_index=True)

        # One-Hot Encoding
        df_encoded = pd.get_dummies(df, drop_first=True)

        # Extract the last row (which is the input data)
        encoded_input = df_encoded.iloc[[-1]]

        # Standardize numerical features using dataset mean & std
        input_data[num_features] = (input_data[num_features] - df[num_features].mean()) / df[num_features].std()

        # Combine processed numerical and categorical features
        processed_data = pd.concat([input_data[num_features].reset_index(drop=True), encoded_input.reset_index(drop=True)], axis=1)

        # Predict price
        prediction = model.predict(processed_data)[0]

        return jsonify({'predicted_price': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
