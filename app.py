import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="",
    layout="wide"
)

st.title(" Stock Price Prediction Using Machine Learning")
st.write("A simple educational project for predicting stock closing prices.")

st.sidebar.header("Project Information")
st.sidebar.info(
    "This project uses historical stock prices and "
    "a Random Forest Regression model."
)

# Load the CSV file directly from your GitHub repository permanently
url = (
    "https://raw.githubusercontent.com/Ebinlitto/stock-price-prediction/main/AAPL.csv"
)

data = pd.read_csv(url)

# Standardize column names to lowercase to prevent bugs
data.columns = data.columns.str.lower()

st.success("Dataset loaded successfully from GitHub!")


# Find Close column
close_column = None

for column in data.columns:
    if column.lower().strip() == "close":
        close_column = column
        break

if close_column is None:
    st.error("Your CSV must contain a column named 'Close'.")
    st.stop()

data[close_column] = pd.to_numeric(
    data[close_column],
    errors="coerce"
)

data = data.dropna()

if len(data) < 20:
    st.error("Dataset must contain at least 20 valid rows.")
    st.stop()

# Display dataset
st.subheader(" Historical Stock Data")
st.dataframe(data.tail(10), use_container_width=True)

# Historical chart
st.subheader(" Closing Price History")
st.line_chart(data[close_column])

# Feature engineering
data["Previous_Close"] = data[close_column].shift(1)
data["Previous_2_Close"] = data[close_column].shift(2)
data["Previous_3_Close"] = data[close_column].shift(3)

data = data.dropna()

X = data[
    [
        "Previous_Close",
        "Previous_2_Close",
        "Previous_3_Close"
    ]
]

y = data[close_column]

# Time-based split
split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Metrics
st.subheader("Model Performance")

col1, col2 = st.columns(2)

col1.metric("Mean Absolute Error", f"{mae:.2f}")
col2.metric("Root Mean Squared Error", f"{rmse:.2f}")

# Actual vs predicted
st.subheader("Actual vs Predicted Prices")

result = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

st.line_chart(result)

# Next-day prediction
latest_data = X.tail(1)

next_prediction = model.predict(latest_data)[0]

st.subheader("Next-Day Price Prediction")

st.success(
    f"Predicted Closing Price: {next_prediction:.2f}"
)

st.warning(
    "This prediction is for educational purposes only "
    "and should not be used as financial advice."
)

st.subheader("📝 Project Conclusion")

st.write(
    "The Random Forest Regression model was trained using "
    "historical closing prices. The system predicts a possible "
    "next-day closing price and displays the results through "
    "an interactive web dashboard."
  )
