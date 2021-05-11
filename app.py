import modify
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
picklee = pickle.load(open('rf_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def value():
    year = float(request.form['year'])
    seller_type = request.form['seller_type']
    km_driven = request.form['km_driven']
    owner_type = request.form['owner_type']
    fuel_type = request.form['fuel_type']
    transmission_type = request.form['transmission_type']
    seats =float(request.form['seats'])
    company = request.form['company']
    model = request.form['model']
    mileage = float(request.form['mileage'])
    engine = float(request.form['engine'])
    max_price = float(request.form['max_price'])
    min_price = float(request.form['min_price'])

    features = [year, seller_type, km_driven, owner_type, fuel_type, transmission_type, seats, company,
    model, mileage, engine, max_price, min_price]

   
    inputt = modify.convert(features)
    output = picklee.predict([inputt])
    
    return str(output)



if __name__ == '__main__':
	app.run(debug=True)