from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the dataset
housing = pd.read_csv("Mumbai House Prices.csv")

# Function to preprocess the data
def preprocess_data(df):
    # Convert price to Lakhs
    df['Price_in_Lakhs'] = df.apply(lambda x: x['price'] * 100 if x['price_unit'] == 'Cr' else x['price'], axis=1)
    
    # Drop unnecessary columns
    df = df.drop(['price', 'price_unit', 'status'], axis=1)
    
    return df

# Preprocess the data
housing = preprocess_data(housing)

# Function to filter data based on location, number of bedrooms, and apartment name
def filter_by_criteria(df, location, bhk, footage):
    filtered_df = df[(df['region'].str.contains(location, case=False)) & 
                     (df['bhk'] == bhk) & 
                     (df['area'].between(footage - 50, footage + 50))]
    return filtered_df

# Function to train linear regression model
def train_linear_regression(X, Y):
    lr_clf = LinearRegression()
    lr_clf.fit(X, Y)
    return lr_clf

# Function to predict price
def predict_price(lr_model, X):
    return lr_model.predict(np.array(X).reshape(1, -1))[0]

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    location = request.form['location']
    bhk = int(request.form['bhk'])
    footage = float(request.form['footage'])

    filtered_data = filter_by_criteria(housing, location, bhk, footage)

    X = filtered_data[['area']]
    Y = filtered_data['Price_in_Lakhs']
    lr_model = train_linear_regression(X, Y)
    predicted_price = predict_price(lr_model, footage).round(3)
    filtered_data['Price_in_Lakhs'] = filtered_data['Price_in_Lakhs'].astype(int)
    return render_template('index.html', location=location, bhk=bhk, footage=footage, predicted_price=predicted_price, similar_apartments=filtered_data)

if __name__ == '__main__':
    app.run(debug=True)

