from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "secret"

# Load models

# 📁 Set the folder path
MODEL_DIR = "./models"  

# ✅ Load each model from that directory
model_refurb = pickle.load(open(os.path.join(MODEL_DIR, "model_refurb.pkl"), "rb"))
model_resale = pickle.load(open(os.path.join(MODEL_DIR, "model_resale.pkl"), "rb"))
model_decision = pickle.load(open(os.path.join(MODEL_DIR, "model_decision.pkl"), "rb"))

# decision_mapping = pickle.load(open(os.path.join(MODEL_DIR, "decision_mapping.pkl"), "rb"))
decision_mapping = {
    0: "Refurbish",
    1: "Liquidate",
    2: "Recycle",
    3: "Donate",
    4: "Keep It"
}

# Session-like storage
last_prediction = {}

#  walmart main website

@app.route("/")
def walmart():
        
    return render_template("walmart.html")

# ------------------------
# PAGE 1 - DETAIL 1
# ------------------------

@app.route("/detail", methods=["GET", "POST"])
def detail1():
    if request.method == "POST":
        last_prediction.clear()
        last_prediction["product"] = request.form.get("product")
        last_prediction["price"] = float(request.form.get("price"))
        return redirect("/reason")
    return render_template("detail1.html")

# ------------------------
# PAGE 2 - REASON
# ------------------------

@app.route("/reason", methods=["GET", "POST"])
def reason2():
    if not last_prediction:
        return redirect("/")
    if request.method == "POST":
        last_prediction["reason"] = request.form.get("reason")
        return redirect("/payment")
    return render_template("reason2.html", data=last_prediction)

# ------------------------
# PAGE 3 - PAYMENT
# ------------------------

@app.route("/payment", methods=["GET", "POST"])
def payment3():
    if not last_prediction:
        return redirect("/")
    if request.method == "POST":
        last_prediction["comments"] = request.form.get("comments")
        last_prediction["refund_option"] = request.form.get("refund_option")
        return redirect("/success")
    return render_template("payment3.html", data=last_prediction)

# ------------------------
# SUCCESS PAGE
# ------------------------

@app.route("/success", methods=["GET"])
def success_page():
    if not last_prediction:
        return redirect("/")
    return render_template("success_page4.html", data=last_prediction)

# ------------------------
# DASHBOARD PAGE
# ------------------------

@app.route("/dashboard")
def dashboard():
    if not last_prediction:
        return redirect("/")
    return render_template("dashboard.html", data=last_prediction)

# ------------------------
# PREDICT ROUTE
# ------------------------

@app.route("/predict", methods=["POST"])
def predict():
    global last_prediction
    if not last_prediction:
        return jsonify({"error": "No return data available"}), 400

    reason_codes = {
        "defective": 0,
        "wrong-item": 1,
        "size-fit": 2,
        "not-needed": 3,
        "quality": 4,
        "other": 5
    }

    return_reason_code = reason_codes.get(last_prediction["reason"], 5)

    category_codes = {
        "Apple iPhone 15 Pro - 128GB Natural Titanium": 0,
        "Samsung 65\" QLED 4K Smart TV": 1,
        "Nike Air Max 270 - Size 10": 2
    }

    category = category_codes.get(last_prediction["product"], 0)

    inbound_shipping_cost = 100
    condition_grade = 0
    hazardous_goods_flag = 0
    co2_saved_refurb_vs_landfill = 3.5

    raw_features = [
        "category",
        "original_price",
        "condition_grade",
        "return_reason_code",
        "inbound_shipping_cost",
        "hazardous_goods_flag",
        "co2_saved_refurb_vs_landfill"
    ]

    raw_values = [
        category,
        last_prediction["price"],
        condition_grade,
        return_reason_code,
        inbound_shipping_cost,
        hazardous_goods_flag,
        co2_saved_refurb_vs_landfill
    ]

    input_raw = pd.DataFrame([raw_values], columns=raw_features)

    refurb_cost = model_refurb.predict(input_raw, predict_disable_shape_check=True)[0]
    resale_value = model_resale.predict(input_raw, predict_disable_shape_check=True)[0]

    refurb_ratio = refurb_cost / last_prediction["price"] if last_prediction["price"] > 0 else 0
    resale_ratio = resale_value / last_prediction["price"] if last_prediction["price"] > 0 else 0
    profit = resale_value - refurb_cost - inbound_shipping_cost

    full_data = {
        "category": category,
        "original_price": last_prediction["price"],
        "condition_grade": condition_grade,
        "return_reason_code": return_reason_code,
        "inbound_shipping_cost": inbound_shipping_cost,
        "hazardous_goods_flag": hazardous_goods_flag,
        "co2_saved_refurb_vs_landfill": co2_saved_refurb_vs_landfill,
        "refurb_ratio": refurb_ratio,
        "resale_ratio": resale_ratio,
        "profit": profit
    }

    model_features = [
        "category",
        "original_price",
        "condition_grade",
        "return_reason_code",
        "inbound_shipping_cost",
        "hazardous_goods_flag",
        "co2_saved_refurb_vs_landfill",
        "refurb_ratio",
        "resale_ratio",
        "profit"
    ]

    input_full = pd.DataFrame([full_data])[model_features]

    decision_label = model_decision.predict(input_full, predict_disable_shape_check=True)[0]
    decision_text = decision_mapping.get(decision_label, "Unknown")

    prediction_data = {
        "predicted_refurb_cost": round(refurb_cost, 2),
        "predicted_resale_value": round(resale_value, 2),
        "profit": round(profit, 2),
        "recommended_decision": decision_text,
        "action_explanation": "The model suggests this action based on predicted profitability."
    }

    last_prediction.update(prediction_data)

    return jsonify(prediction_data)

if __name__ == "__main__":
    app.run(debug=True)
