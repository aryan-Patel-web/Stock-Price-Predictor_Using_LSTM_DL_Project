from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model("saved_lstm_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load the predictions CSV
df = pd.read_csv("predictions.csv")

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_price = None
    if request.method == "POST":
        try:
            user_input = float(request.form["close_price"])

            # Get last 59 prices and append user input to make 60
            last_60 = df["Close"].values[-59:]
            sequence = np.append(last_60, user_input).reshape(-1, 1)

            # Scale and reshape for LSTM input
            scaled = scaler.transform(sequence)
            lstm_input = np.reshape(scaled, (1, 60, 1))

            # Predict and inverse transform
            prediction = model.predict(lstm_input)
            predicted_price = scaler.inverse_transform(prediction)[0][0]

        except Exception as e:
            predicted_price = f"Error: {str(e)}"

    return render_template("index.html",
                           dates=list(range(len(df))),
                           actual=df["Close"].tolist(),
                           predicted=df["Predictions"].tolist(),
                           predicted_price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)
