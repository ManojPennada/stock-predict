import yfinance as yf
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

app = Flask(__name__)

# Function to retrieve stock data for a given ticker symbol and date range


def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to preprocess stock data and prepare features and target variable


def preprocess_data(stock_data):
    # We'll use the 'Close' prices as our feature
    # Exclude the last row for prediction
    features = stock_data['Close'].values[:-1]

    # Shift the 'Close' prices by one day to use as the target variable
    target = stock_data['Close'].shift(-1).dropna().values

    return features, target

# Function to train the Random Forest model


def train_random_forest(features, target):
    if len(features) < 2:
        print("Insufficient data for training.")
        return None

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42)

    # Initialize the Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    rf_regressor.fit(X_train.reshape(-1, 1), y_train)

    return rf_regressor

# Function to train the XGBoost model


def train_xgboost(features, target):
    if len(features) < 2:
        print("Insufficient data for training.")
        return None

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42)

    # Initialize the XGBoost Regressor
    xgb_regressor = XGBRegressor()

    # Train the model
    xgb_regressor.fit(X_train.reshape(-1, 1), y_train)

    return xgb_regressor

# Function to predict the next closing price using the trained model


def predict_next_closing_price(model, latest_price):
    next_closing_price = model.predict(np.array([latest_price]).reshape(-1, 1))
    return next_closing_price[0]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        model_type = request.form['model']

        # Get stock data
        stock_data = get_stock_data(
            ticker, start_date="2024-04-01", end_date="2024-05-11")

        # Preprocess the data
        features, target = preprocess_data(stock_data)

        # Train the model
        if model_type == 'random_forest':
            model = train_random_forest(features, target)
        elif model_type == 'xgboost':
            model = train_xgboost(features, target)

        if model is not None:
            # Predict the next closing price
            latest_price = stock_data['Close'].iloc[-1]
            next_closing_price = predict_next_closing_price(
                model, latest_price)
            return render_template('index.html', prediction=next_closing_price)

    return render_template('index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
