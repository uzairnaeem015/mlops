#Import Libraries
import pickle
from flask import Flask

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
    data_file_path = '<<your-bucket-name>>'
    df = pd.read_csv(data_file_path)

  # Features and Labels
    X = df[['rooms','sqft']].values  # Features
    y = df['price'].values           # Label

  #Split the data for Training and Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,random_state=0)

    # Build the model
    lr = LinearRegression()
    model = lr.fit(X_train, y_train)

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

if __name__ == "__main__":
    app.run('0.0.0.0',5000)