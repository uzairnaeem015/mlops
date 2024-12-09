# Import essential Libraries
import numpy as np
import pandas as pd
import logging
import boto3
from botocore.exceptions import ClientError
import os
import pickle
from flask import Flask, jsonify, request
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Environment Variables for AWS Credentials
bucket = 'test123-un-east-2'
data_file_name = 'data/rental_1000.csv'
object_name = "model.pkl"
file_name = "model-rental-prediction.pkl"

# Create a session using your AWS credentials from environment variables
session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name='us-east-2'  # Change to your desired region
)

s3_client = session.client('s3')
model = None  # Global variable to store the loaded model

@app.route("/upload_model", methods=['POST'])
def upload_model():
    global model
    try:
        if model is not None:
            # Save the model using pickle
            pickle.dump(model, open(file_name, 'wb'))
            s3_client.upload_file(file_name, bucket, object_name)
            return jsonify({"message": "Model developed and uploaded successfully."}), 200
        else:
            return jsonify({"error": "Model is not developed yet."}), 400
    except ClientError as e:
        logging.error(e)
        return jsonify({"error": "Failed to upload the model."}), 500

@app.route("/load_model", methods=['POST'])
def load_model():
    global model
    try:
        s3_client.download_file(bucket, object_name, file_name)
        model = pickle.load(open(file_name, 'rb'))
        return jsonify({"message": "Model loaded successfully."}), 200
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return jsonify({"error": "Failed to load the model."}), 500

@app.route("/develop_model", methods=['POST'])   
def develop_model():
    try:
        
        # Upload the file
        s3_client = boto3.client('s3')
        try:
            response = s3_client.upload_file(data_file_name, bucket, "rental_1000.csv")
        except ClientError as e:
            logging.error(e)
            return False

        rentalDF = pd.read_csv(f"s3://{bucket}/rental_1000.csv")  # For AWS Cloud

        # Consider Features and Labels
        X = rentalDF[["rooms", "sqft"]].values  # Features
        y = rentalDF["price"].values  # Labels

        # Split the Training Data and Testing Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        # Training the Dataset
        global model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Save the model to S3
        if upload_model():
            return jsonify({"message": "Model developed and uploaded successfully."}), 200
        else:
            return jsonify({"error": "Failed to upload the model."}), 500
        
    except Exception as e:
        logging.error(f"Error developing model: {e}")
        return jsonify({"error": "Failed to develop the model."}), 500

@app.route("/predict", methods=['POST'])
def predict():
    user_input = request.json

    rooms = int(user_input.get('rooms', 0))
    area = int(user_input.get('area', 0))
    input_data = np.array([[rooms, area]])

    if model is None:
        load_model()
        # return jsonify({"error": "Model is not loaded. Please develop the model first."}), 500

    try:
        predict_rental_price = model.predict(input_data)
        return jsonify({"Predicted Rental Price Using Developed Model": predict_rental_price[0]}), 200
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Failed to predict the rental price."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
