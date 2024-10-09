This project aims to forecast stock prices using various statistical and deep learning models such as ARIMA, GRU, LSTM, and RNN. The data for this project is fetched from Yahoo Finance using the yfinance library and consists of daily stock prices.


Data Description

The dataset consists of daily stock prices, including the open, high, low, and close prices, and volume, fetched using finance. The closing prices are used to train and test the models.

Techniques and Models

ARIMA (AutoRegressive Integrated Moving Average)

Stationarity Test: I applied the Dickey-Fuller Test to check whether the time series data is stationary. Non-stationary data was transformed using first-order differencing.
Model Selection: Based on autocorrelation and partial autocorrelation plots, I identified the parameters for the ARIMA model.

ARIMA Model: I used an ARIMA(1,1,1) model to forecast stock prices.

 Deep Learning Method - I used the closing prices of the stock, split it into training and test sets, and scaled the data before applying deep learning models.

Conclusion
From the results, it is evident that the LSTM model performs the best, with a Mean Squared Error (MSE) of 6.41 and an R2 value of 0.94. This model can be further applied to forecast other stock prices and potentially generate profitable insights.

