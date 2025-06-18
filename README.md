# 📈 LSTM Stock Price Predictor

A machine learning project that predicts stock closing prices using an LSTM (Long Short-Term Memory) neural network. The project includes training the model with historical data, scaling it with MinMaxScaler, and deploying it as a Flask web app with interactive Chart.js visualizations.

---

## 🔧 Technologies Used

- Python
- TensorFlow / Keras (LSTM Model)
- NumPy, Pandas (Data Processing)
- Scikit-learn (MinMaxScaler)
- Flask (Web App Backend)
- HTML, CSS, Chart.js (Frontend Visualization)
- Jinja2 (Template Rendering)
- Pickle (Saving Scaler)
- HDF5 (`.h5`) for Model Storage

---
📂 Project Structure

# lstm-stock-price-predictor/
│
├── model_training/
│   ├── train_model.py         # Preprocess data, build & train LSTM, save model
│   ├── saved_lstm_model.h5    # Trained LSTM model file
│   ├── scaler.pkl             # Saved MinMaxScaler for inverse transform
│   └── predictions.csv        # Actual vs predicted closing prices
│
├── web_app/
│   ├── app.py                 # Flask backend to load model and serve web UI
│   ├── templates/
│     └── index.html         # HTML + Chart.js frontend
│   
│                 
│
├── README.md                  # Full project documentation
└── requirements.txt           # All Python package dependencies


Install Dependencies
List these in your requirements.txt:

tensorflow
keras
pandas
numpy
matplotlib
scikit-learn
flask



## 🚀 How to Run the App

1. **Clone the repo**  
```bash
git clone https://github.com/aryan-Patel-web/lstm-stock-price-predictor.git
cd lstm-stock-price-predictor/web_app

📊 Features ---------------
📈 Plot actual vs predicted stock closing prices

🧮 Predict next closing price from recent sequence

💾 Uses a pre-trained LSTM model (saved in .h5)

🔄 Uses same scaler (scaler.pkl) for prediction as training

🧪 Sample Examples

Last known 59 prices: [204.5, 205.6, ..., 209.4]
User input Close: 209.4
Model Output: 210.03 (predicted next close)

Last known 59 prices: [250.1, 252.6, ..., 260.0]
User input Close: 260.0
Model Output: 215.2 ❌ (Incorrect due to scaling error or model drift)

