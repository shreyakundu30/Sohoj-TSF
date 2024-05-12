from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import statistics
import os
from sktime.forecasting.ets import AutoETS
from sktime.utils.plotting import plot_series
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend to avoid GUI-related issues
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from flask import Flask, render_template, request, redirect, url_for
import plotly.express as px
from io import BytesIO
import base64
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import yfinance as yf
import uuid
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from math import sqrt
from flask_mail import Mail, Message
from statsmodels.tsa.seasonal import seasonal_decompose
import os
from datetime import datetime, timedelta
import plotly.graph_objs as go
from pmdarima.arima import auto_arima
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

def auto_ets(values, seasonal_periods=12):
    best_aic = float("inf")
    best_ets_model = None

    for trend in ['add', 'mul', None]:
        for seasonal in ['add', 'mul', None]:
            try:
                model = ETSModel(values, error='add', trend=trend, damped_trend=False, seasonal=seasonal, seasonal_periods=seasonal_periods)
                fitted_model = model.fit()
                aic = fitted_model.aic
                if aic < best_aic:
                    best_aic = aic
                    best_ets_model = fitted_model
            except:
                continue

    if best_ets_model is None:
        raise ValueError("Unable to fit ETS model")

    return best_ets_model

class NaiveModel:
    def __init__(self):
        pass

    def fit(self, train_data):
        # Naive model doesn't require fitting, just return the data
        return train_data

    def forecast(self, steps):
        # Naive model forecasts by simply repeating the last observed value
        return [self.last_observed_value] * steps
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = "847b5600-f639-4d07-803e-de716f4e89b7"
app.config.update(
        MAIL_SERVER = 'smtp.gmail.com',
        MAIL_PORT = '465',
        MAIL_USE_SSL = True,
        MAIL_USE_TLS = False,
        MAIL_USERNAME = "sohojtsf@gmail.com",
        MAIL_PASSWORD = "ulwvcmtlzpxylhvh",
        MAIL_DEFAULT_SENDER = "sohojtsf@gmail.com"
)

mail = Mail(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def perform_forecast(values, model_type, horizon, frequency):
    if model_type == 'auto_arima':
        try:
            model = auto_arima(values, seasonal=True, m=12)
        except ValueError:
            model = auto_arima(values, seasonal=False)
        forecast = model.predict(n_periods=horizon)
    elif model_type == 'auto_ets':
        model = ExponentialSmoothing(values, trend='add', seasonal='add', seasonal_periods=12).fit()
        forecast = model.forecast(steps=horizon)
    elif model_type == 'ses':
        model = ExponentialSmoothing(values, trend='add', seasonal='add', seasonal_periods=12).fit()
        forecast = model.forecast(steps=horizon)
    elif model_type == 'des':
        model = ExponentialSmoothing(values, trend='add', seasonal='add', seasonal_periods=12, damped=True).fit()
        forecast = model.forecast(steps=horizon)
    elif model_type == 'tes':
        model = ExponentialSmoothing(values, trend='add', seasonal='add', seasonal_periods=12, damped_trend=True).fit()
        forecast = model.forecast(steps=horizon)
    elif model_type == 'naive':
        # Naive forecasting: Forecast each future value to be the same as the value observed at the corresponding time in the past
        forecast = values.iloc[-horizon:].values
    else:
        raise ValueError("Invalid model type")
    
    return forecast

def generate_dates(end_date, horizon, frequency):
    if frequency == 'daily':
        dates = pd.date_range(start=end_date + timedelta(days=1), periods=horizon, freq='D')
    elif frequency == 'monthly':
        dates = pd.date_range(start=end_date + timedelta(days=1), periods=horizon, freq='MS')
    elif frequency == 'yearly':
        dates = pd.date_range(start=end_date + timedelta(days=1), periods=horizon, freq='YS')
    else:
        raise ValueError("Invalid frequency")
    return dates

@app.route('/csv', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        selected_model = request.form['model']
        horizon = int(request.form['horizon'])
        frequency = request.form['frequency']
        
        # Save the uploaded file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        df = load_data(file_path)
        values = df['values']
        forecast_values = perform_forecast(values, selected_model, horizon, frequency)
        
        # Generate dates for the forecasted period
        end_date = pd.to_datetime(df['dates'].iloc[-1])
        forecast_dates = generate_dates(end_date, horizon, frequency)
        
        # Plot the current data
        fig_current = go.Figure()
        fig_current.add_trace(go.Scatter(x=df['dates'], y=values, mode='lines+markers', name='Current Data'))
        fig_current.update_layout(title='Current Data', xaxis_title='Date', yaxis_title='Value')
        current_graph = fig_current.to_html(full_html=False)
        
        # Plot the forecasted data
        fig_forecasted = go.Figure()
        fig_forecasted.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, mode='lines+markers', name='Forecasted Data'))
        fig_forecasted.update_layout(title='Forecasted Data', xaxis_title='Date', yaxis_title='Value')
        forecasted_graph = fig_forecasted.to_html(full_html=False)
        
        # Determine forecast horizon text
        if frequency == 'daily':
            horizon_text = f"Prediction for {horizon} days"
        elif frequency == 'monthly':
            horizon_text = f"Prediction for {horizon} months"
        elif frequency == 'yearly':
            horizon_text = f"Prediction for {horizon} years"
        else:
            horizon_text = "Unknown Forecast Horizon"
        
        # Customize model names for display
        model_names = {
            'auto_arima': 'Auto ARIMA',
            'auto_ets': 'Auto ETS',
            'ses': 'SES',
            'des': 'DES',
            'tes': 'TES',
            'naive': 'Naive'
        }
        
        # Pass the customized model name to the template
        model_name = model_names.get(selected_model, 'Unknown Model')
        
        return render_template('prediction.html', current_graph=current_graph, forecasted_graph=forecasted_graph, horizon_text=horizon_text, model_name=model_name)

    return render_template('index.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/contact',methods=['GET','POST'])
def contact():
    if (request.method=='POST'):
        First_name=request.form.get('First_Name')
        Last_name=request.form.get('Last_Name')
        email=request.form.get('email')
        mob=request.form.get('mob')
        message=request.form.get('message')
        email_msg = Message(subject="Your Message Received: সহজ- TSF to the Rescue!",recipients=[email],
        body=f"Hello {First_name} {Last_name} ,\n\nThank you for choosing সহজ- TSF for your stock price prediction needs. We value your interest and want to assure you that our dedicated team is actively working to address your queries. Your satisfaction is our top priority, and we are committed to providing you with accurate information and assistance promptly.\n\nWarm regards,\nThe সহজ- TSF Team",
        )
        admin_msg = Message(
        subject=f"New Contact: {First_name} {Last_name}",
        recipients=['hritwikasaha2003@gmail.com','shreyakundu004@gmail.com'],
        body=f"{First_name} {Last_name} has contacted.\n\n Email: {email},\n Mobile: {mob},\n Message: {message}"
    )

        mail.send(email_msg)
        mail.send(admin_msg)		
    return render_template('contact.html')


def train_test_split(data, train_percent, test_percent):
    train_size = int(len(data) * (train_percent / 100))
    train_data = data[:train_size]
    test_data = data[train_size:train_size+int(len(data) * (test_percent / 100))]
    return train_data, test_data

def evaluate_forecast(test_data, forecast):
    rmse = sqrt(mean_squared_error(test_data, forecast))
    mape = (abs(test_data - forecast) / test_data).mean() * 100
    return rmse, mape

def generate_forecasts(df, train_percent, test_percent):
    data_column = df.columns[1]  # Assuming 'values' column is the second column

    data = df[data_column]

    train_data, test_data = train_test_split(data, train_percent, test_percent)

    results = {}


    # Auto ARIMA
    arima_uuid = str(uuid.uuid4())
    arima_graph_path = f'static/{arima_uuid}.png'
    model = ARIMA(train_data, order=(1, 1, 1))
    fit = model.fit()
    forecast = fit.forecast(steps=len(test_data))
    arima_rmse, arima_mape = evaluate_forecast(test_data, forecast)
    train_forecast = fit.fittedvalues
    train_rmse, train_mape = evaluate_forecast(train_data, train_forecast)
    results['ARIMA'] = {'train_rmse': f'{train_rmse:.2f}', 'test_rmse': f'{arima_rmse:.2f}', 'train_mape': f'{train_mape:.2f}', 'test_mape': f'{arima_mape:.2f}', 'graph': arima_graph_path}
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data, label='Training Data')
    plt.plot(test_data.index, test_data, label='Test Data')
    plt.plot(test_data.index, forecast, label='Forecast')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Auto ARIMA Forecast')
    plt.legend()
    plt.savefig(arima_graph_path)
    plt.close()

    # Auto ETS
    ets_uuid = str(uuid.uuid4())
    ets_graph_path = f'static/{ets_uuid}.png'
    model = ETSModel(train_data, error="add", trend="add", seasonal="add", seasonal_periods=12)
    fit = model.fit()
    forecast = fit.forecast(steps=len(test_data))
    ets_rmse, ets_mape = evaluate_forecast(test_data, forecast)
    train_forecast = fit.fittedvalues
    train_rmse, train_mape = evaluate_forecast(train_data, train_forecast)
    results['Auto ETS'] = {'train_rmse': f'{train_rmse:.2f}', 'test_rmse': f'{ets_rmse:.2f}', 'train_mape': f'{train_mape:.2f}', 'test_mape': f'{ets_mape:.2f}', 'graph': ets_graph_path}
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data, label='Training Data')
    plt.plot(test_data.index, test_data, label='Test Data')
    plt.plot(test_data.index, forecast, label='Forecast')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Auto ETS Forecast')
    plt.legend()
    plt.savefig(ets_graph_path)
    plt.close()

    # Simple Exponential Smoothing (SES)
    ses_uuid = str(uuid.uuid4())
    ses_graph_path = f'static/{ses_uuid}.png'
    model = SimpleExpSmoothing(train_data)
    fit = model.fit()
    forecast = fit.forecast(len(test_data))
    ses_rmse, ses_mape = evaluate_forecast(test_data, forecast)
    train_forecast = fit.fittedvalues
    train_rmse, train_mape = evaluate_forecast(train_data, train_forecast)
    results['SES'] = {'train_rmse': f'{train_rmse:.2f}', 'test_rmse': f'{ses_rmse:.2f}', 'train_mape': f'{train_mape:.2f}', 'test_mape': f'{ses_mape:.2f}', 'graph': ses_graph_path}
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data, label='Training Data')
    plt.plot(test_data.index, test_data, label='Test Data')
    plt.plot(test_data.index, forecast, label='Forecast')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Simple Exponential Smoothing (SES) Forecast')
    plt.legend()
    plt.savefig(ses_graph_path)
    plt.close()

    # Holt's Exponential Smoothing (DES)
    des_uuid = str(uuid.uuid4())
    des_graph_path = f'static/{des_uuid}.png'
    model = Holt(train_data)
    fit = model.fit()
    forecast = fit.forecast(len(test_data))
    des_rmse, des_mape = evaluate_forecast(test_data, forecast)
    train_forecast = fit.fittedvalues
    train_rmse, train_mape = evaluate_forecast(train_data, train_forecast)
    results['DES'] = {'train_rmse': f'{train_rmse:.2f}', 'test_rmse': f'{des_rmse:.2f}', 'train_mape': f'{train_mape:.2f}', 'test_mape': f'{des_mape:.2f}', 'graph': des_graph_path}
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data, label='Training Data')
    plt.plot(test_data.index, test_data, label='Test Data')
    plt.plot(test_data.index, forecast, label='Forecast')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title("Holt's Exponential Smoothing (DES) Forecast")
    plt.legend()
    plt.savefig(des_graph_path)
    plt.close()


    # Triple Exponential Smoothing (TES)
    tes_uuid = str(uuid.uuid4())
    tes_graph_path = f'static/{tes_uuid}.png'
    model = ExponentialSmoothing(train_data, seasonal_periods=12, trend='add', seasonal='add')
    fit = model.fit()
    forecast = fit.forecast(len(test_data))
    tes_rmse, tes_mape = evaluate_forecast(test_data, forecast)
    train_forecast = fit.fittedvalues
    train_rmse, train_mape = evaluate_forecast(train_data, train_forecast)
    results['TES'] = {'train_rmse': f'{train_rmse:.2f}', 'test_rmse': f'{tes_rmse:.2f}', 'train_mape': f'{train_mape:.2f}', 'test_mape': f'{tes_mape:.2f}', 'graph': tes_graph_path}
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data, label='Training Data')
    plt.plot(test_data.index, test_data, label='Test Data')
    plt.plot(test_data.index, forecast, label='Forecast')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Triple Exponential Smoothing (TES) Forecast')
    plt.legend()
    plt.savefig(tes_graph_path)
    plt.close()

    #naive model
    naive_uuid = str(uuid.uuid4())
    naive_graph_path = f'static/{naive_uuid}.png'
    naive_model = NaiveModel()
    naive_model.last_observed_value = train_data.iloc[-1]  # Set the last observed value
    forecast = naive_model.forecast(len(test_data))
    naive_rmse, naive_mape = evaluate_forecast(test_data, forecast)
    train_forecast = [train_data.iloc[-1]] * len(train_data)
    train_rmse, train_mape = evaluate_forecast(train_data, train_forecast)
    results['Naive'] = {'train_rmse': f'{train_rmse:.2f}', 'test_rmse': f'{naive_rmse:.2f}', 'train_mape': f'{train_mape:.2f}', 'test_mape': f'{naive_mape:.2f}', 'graph': naive_graph_path}
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data, label='Training Data')
    plt.plot(test_data.index, test_data, label='Test Data')
    plt.plot(test_data.index, forecast, label='Forecast')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Naive Forecast')
    plt.legend()
    plt.savefig(naive_graph_path)
    plt.close()

    return results

@app.route('/ind')
def ind():
    return render_template('in.html')

@app.route('/plot', methods=['POST'])
def plot():
    file = request.files['file']
    train_percent = int(request.form['train_percent'])
    test_percent = int(request.form['test_percent'])

    file_path = f"uploads/{file.filename}"
    file.save(file_path)

    df = pd.read_csv(file_path)

    results = generate_forecasts(df, train_percent, test_percent)

    os.remove(file_path)

    return render_template('re.html',
    results=results)

def get_stock_data(company_name, period):
    ticker = yf.Ticker(company_name)
    if period == 'days':
        data = ticker.history(period='7d')
    elif period == 'months':
        data = ticker.history(period='1mo')
    elif period == 'years':
        data = ticker.history(period='1y')
    else:
        return None
    return data

def plot_graph(data, predicted_data=None):
    plt.figure(figsize=(10, 6))
    
    # Plot original data in blue color
    plt.plot(data.index, data['Close'], label='Original Data', color='blue')
    
    # Plot predicted data in red color if available
    if predicted_data is not None:
        plt.plot(predicted_data.index, predicted_data['Close'], label='Predicted Data', color='red', linestyle='--')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.grid(True)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png).decode()
    buffer.close()
    return graph

def seasonal_naive_forecast(data, period=7, steps=7):
    forecast = []
    data_len = len(data)
    for i in range(steps):
        forecast.append(data.iloc[(data_len - period + i) % data_len]['Close'])
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    return pd.DataFrame({'Close': forecast}, index=forecast_index)

@app.route('/ticker', methods=['GET', 'POST'])
def ticker():
    if request.method == 'POST':
        company_name = request.form['company']
        period = request.form['period']
        data = get_stock_data(company_name, period)
        if data is None:
            return render_template('ticker.html', error='Invalid period selected!')
        else:
            predicted_data = seasonal_naive_forecast(data, steps=730)  # 2 years = 730 days
            graph = plot_graph(data, predicted_data)
            graph_data = 'data:image/png;base64,' + graph
            return render_template('ticker.html', graph_data=graph_data)
    return render_template('ticker.html')   

if __name__ == '__main__':
    app.run(debug=True)
