from flask import Flask, render_template, request
import joblib
import numpy as np
import pymysql

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

# MySQL connection function
def get_db_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="123456",
        database="xyz_holidays"
    )

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    # Collect inputs
    Age = int(request.form["Age"])
    TypeofContact = int(request.form["TypeofContact"])
    CityTier = int(request.form["CityTier"])
    DurationOfPitch = int(request.form["DurationOfPitch"])
    Occupation = int(request.form["Occupation"])
    Gender = int(request.form["Gender"])
    NumberOfPersonVisiting = int(request.form["NumberOfPersonVisiting"])
    NumberOfFollowups = int(request.form["NumberOfFollowups"])
    ProductPitched = int(request.form["ProductPitched"])
    PreferredPropertyStar = int(request.form["PreferredPropertyStar"])
    MaritalStatus = int(request.form["MaritalStatus"])
    NumberOfTrips = int(request.form["NumberOfTrips"])
    Passport = int(request.form["Passport"])
    PitchSatisfactionScore = int(request.form["PitchSatisfactionScore"])
    OwnCar = int(request.form["OwnCar"])
    NumberOfChildrenVisiting = int(request.form["NumberOfChildrenVisiting"])
    Designation = int(request.form["Designation"])
    MonthlyIncome = int(request.form["MonthlyIncome"])

    features = np.array([
        Age, TypeofContact, CityTier, DurationOfPitch, Occupation,
        Gender, NumberOfPersonVisiting, NumberOfFollowups, ProductPitched,
        PreferredPropertyStar, MaritalStatus, NumberOfTrips, Passport,
        PitchSatisfactionScore, OwnCar, NumberOfChildrenVisiting,
        Designation, MonthlyIncome
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]
    prediction_label = "Customer WILL take the product" if prediction == 1 else "Customer will NOT take the product"

    # 🔹 Save to MySQL
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO travel_predictions (
        Age, TypeofContact, CityTier, DurationOfPitch, Occupation,
        Gender, NumberOfPersonVisiting, NumberOfFollowups, ProductPitched,
        PreferredPropertyStar, MaritalStatus, NumberOfTrips, Passport,
        PitchSatisfactionScore, OwnCar, NumberOfChildrenVisiting,
        Designation, MonthlyIncome, Prediction ,PredictionLabel
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    values = (
        Age, TypeofContact, CityTier, DurationOfPitch, Occupation,
        Gender, NumberOfPersonVisiting, NumberOfFollowups, ProductPitched,
        PreferredPropertyStar, MaritalStatus, NumberOfTrips, Passport,
        PitchSatisfactionScore, OwnCar, NumberOfChildrenVisiting,
        Designation, MonthlyIncome, prediction, prediction_label
    )

    cursor.execute(query, values)
    conn.commit()
    cursor.close()
    conn.close()

    return render_template("index.html", prediction=prediction_label)

if __name__ == "__main__":
    app.run(debug=True)
