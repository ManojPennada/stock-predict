import yfinance as yf
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

app = Flask(__name__)


def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data


def preprocess_data(stock_data):

    features = stock_data['Close'].values[:-1]

    target = stock_data['Close'].shift(-1).dropna().values

    return features, target


def train_random_forest(features, target):
    if len(features) < 2:
        print("Insufficient data for training.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42)

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    rf_regressor.fit(X_train.reshape(-1, 1), y_train)

    return rf_regressor


def train_xgboost(features, target):
    if len(features) < 2:
        print("Insufficient data for training.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42)

    xgb_regressor = XGBRegressor()

    xgb_regressor.fit(X_train.reshape(-1, 1), y_train)

    return xgb_regressor


def predict_next_closing_price(model, latest_price):
    next_closing_price = model.predict(np.array([latest_price]).reshape(-1, 1))
    return next_closing_price[0]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        model_type = request.form['model']

        stock_data = get_stock_data(
            ticker, start_date=start_date, end_date=end_date)

        features, target = preprocess_data(stock_data)

        if model_type == 'random_forest':
            model = train_random_forest(features, target)
        elif model_type == 'xgboost':
            model = train_xgboost(features, target)

        if model is not None:

            latest_price = stock_data['Close'].iloc[-1]
            next_closing_price = predict_next_closing_price(
                model, latest_price)
            return render_template('index.html', prediction=next_closing_price)

    return render_template('index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
