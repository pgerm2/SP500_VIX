import pandas as pd
import statsmodels.tsa.stattools as ts
import statsmodels.tsa.arima_model as arima

# Load historical data for SPX and VIX
data = pd.read_csv('spx_vix_data.csv')

# Check stationarity using the Augmented Dickey-Fuller test
adf_result = ts.adfuller(data['SPX'])
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])

# If necessary, apply differencing to make the series stationary
if adf_result[1] > 0.05:
    data['SPX_diff'] = data['SPX'].diff().dropna()

# Fit an ARIMA model
model = arima.ARIMA(data['SPX_diff'], order=(1, 0, 1)).fit()
print(model.summary())
