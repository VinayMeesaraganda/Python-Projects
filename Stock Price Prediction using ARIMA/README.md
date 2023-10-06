# Stock Price Prediction using ARIMA

This project builds an ARIMA model to predict stock prices using Apple's historical stock price data.

## Data

The historical Apple stock price data is downloaded using the yfinance library in Python. The data covers a period of 5 years. 

The closing prices are extracted and resampled to be monthly frequency for modeling.

## Methods

The steps involved are:

- Check for stationarity and take first difference to make the time series stationary
- Plot ACF and PACF graphs to determine p, d, q parameters  
- Split data into training and validation sets
- Grid search to find the optimal p, d, q parameters based on lowest RMSE
- Fit ARIMA model on training data with selected parameters
- Make predictions on validation data and calculate RMSE
- Refit model on entire data and make predictions for future dates

The ARIMA model with parameters (4,1,4) was found to produce the lowest RMSE.

## Results

The final model is able to make reasonably accurate monthly predictions for the Apple stock price. The predictions closely follow the actual values.

The Jupyter Notebook contains the full code and analysis. 

## Usage

To use this project, simply run the Jupyter Notebook after installing the required libraries. The main libraries used are:

- yfinance
- pandas
- statsmodels
- pmdarima
- matplotlib

The notebook contains detailed explanations and visualizations for each step.
