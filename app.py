from flask import Flask,render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle


app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        loc = request.form.get("Location")
        bed = request.form.get("bedrooms")
        bath = request.form.get("bathrooms")
        sqft = request.form.get("size")
        con = request.form.get("condition")
        year = request.form.get("year_built")
        i_d = []
        i_d.append(bed)
        i_d.append(bath)
        i_d.append(sqft)
        i_d.append(con)
        i_d.append(year)
        loc1 = int(loc[-2:])
        for i in range(3,75):
            if loc1 != i:
                i_d.append(0.0)
            else:
                i_d.append(1.0)
        loaded_model = pickle.load(open("house_price.pk1", 'rb'))
        price = loaded_model.predict(i_d)
        return render_template("result.html", predicted=price)
    else:
        return render_template("index.html")