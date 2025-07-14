from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your trained model
model = pickle.load(open("return_mode_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    
    if request.method == "POST":

        resale_value = float(request.form["resale_value"])
        price = float(request.form["price"])


        input_data = pd.DataFrame([[resale_value, price]],
                          columns=["resale_value", "price"])
        
        pred = model.predict(input_data)

        prediction = pred[0]
    
    return render_template("index.html", prediction=prediction)



if __name__ == "__main__":
    app.run(debug=True)

