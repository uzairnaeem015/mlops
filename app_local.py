#Import Libraries
import pickle
from flask import Flask
import os
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

app = Flask(__name__)

@app.route("/train")
def train():
  # Load the Dataset
    df = pd.read_csv('data/rental_1000.csv')

  # Features and Labels
    X = df[['rooms','sqft']].values  # Features
    y = df['price'].values           # Label

  #Split the data for Training and Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,random_state=0)

    # Build the model
    lr = LinearRegression()
    model = lr.fit(X_train, y_train)

    os.makedirs(os.path.dirname('model/'), exist_ok=True)
    # Use Pickle to dump the Model
    with open('model/rental-prediction-model.pkl', 'wb') as f:
        pickle.dump(model,f)
    
    # Root Mean Square Error and Score Checks
    y_pred = model.predict(X_test)
    root_mean_squared_error_score = root_mean_squared_error(y_test, y_pred)
    r2_score_value = r2_score(y_test, y_pred)

    print("####################################### Started Model Development #############################################")
    print("Root Mean Squared Error for Built Model:",root_mean_squared_error_score + "Accuary Score for Built Model:",r2_score_value)
    print("Accuary Score for Built Model",r2_score_value)
    print("####################################### Concluded Model Development #######################################")


@app.route("/predict")
def predict():
    # Use Pickle to dump the Model
    with open('model/rental-prediction-model.pkl', 'rb') as f:
      model = pickle.load(f)

    # Using Data as JSON File
    with open('inputs/inputs.json', 'r') as f:
      user_input = json.load(f)

    # User Entries Using JSON
    rooms = user_input['rooms']
    sqft = user_input['sqft']
    user_input_prediction= np.array([[rooms,sqft]])

    # Model Predicition
    predicted_rental_price = model.predict(user_input_prediction)
    output = {"Rental Prediction using Built Model": predicted_rental_price[0]}

    

    with open('outputs/outputs.json', 'w') as f:
      json.dump(output, f)



    print("####################################### Prediction Started Using Model #######################################")
    print("Model Predicted and Results are uploaded to outputs")
    print("####################################### Prediction Concluded Using Model #######################################")

    return output

if __name__ == "__main__":
    app.run('0.0.0.0',5000)
