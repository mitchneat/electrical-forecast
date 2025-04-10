import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import scipy.stats as stats
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, RangeTool, Range1d
from bokeh.io import show
from bokeh.layouts import column
from datetime import timedelta

'''
#############################
############ READ IN DATA and group by day ######################
#############################
'''
df = pd.read_csv("./Cleaned_Data/cleaned_home_energy_usage.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.set_index("Timestamp")


df_daily = df.resample("D").sum()
df_weekly = df.resample("W").sum()
print(df_daily.head())


'''
#############################
############ Plot daily data over time (time series)
#############################
'''
plt.figure(figsize=(14, 6))
plt.plot(df_daily.index, df_daily['kWh'], label="Daily kWh Usage", color='seagreen', alpha=0.6)
# add 7 day rolling average 
df_daily['rolling_mean'] = df_daily['kWh'].rolling(window=7, min_periods=1).mean()
plt.plot(df_daily.index, df_daily['rolling_mean'], label="7-Day Moving Avg", color='red', linewidth=2)
# add grey for weekends
for date in df_daily.index:
    if date.weekday() in [5, 6]:  # Saturday or Sunday
        plt.axvspan(date - pd.Timedelta(hours=12), date + pd.Timedelta(hours=12), color='gray', alpha=0.1)
plt.xlabel("Date")
plt.ylabel("Daily kWh Usage")
plt.title("Energy Usage Over Time")
plt.legend()
plt.grid(alpha=0.3)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
plt.tight_layout() 
# plt.show()
# plt.close()


'''
#############################
############# bar time series 
#############################
'''

# Create a figure
plt.figure(figsize=(14, 6))

plt.bar(df_daily.index, df_daily['kWh'], label="Daily kWh Usage", color='seagreen', alpha=0.6)
# Add 7-day moving average as a red line
df_daily['rolling_avg'] = df_daily['kWh'].rolling(window=7, min_periods=1).mean()
plt.plot(df_daily.index, df_daily['rolling_avg'], label="7-Day Moving Avg", color='red', linewidth=2)
# Add grey shading for weekends
for date in df_daily.index:
    if date.weekday() >= 5:  # Saturday (5) or Sunday (6)
        plt.axvspan(date - pd.Timedelta(hours=12), date + pd.Timedelta(hours=12), color='gray', alpha=0.1)
# Formatting
plt.xlabel("Date")
plt.ylabel("Daily kWh Usage")
plt.title("Daily Electricity Usage Over Time")
plt.legend()
plt.grid(axis='y', alpha=0.3)
# Improve x-axis formatting
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
# Show the plot
# plt.show()


'''
#############################
############# bar plot with range selection 
#############################
'''
df_bokeh = df_daily.copy()
df_bokeh['date'] = pd.to_datetime(df_bokeh.index)
source = ColumnDataSource(data=dict(date=df_bokeh['date'], kWh=df_bokeh['kWh']))
bar_width = timedelta(days=0.8).total_seconds() * 1000 
# get limits so I can increase the upper limit of the graph
y_min, y_max = df_bokeh['kWh'].min(), df_bokeh['kWh'].max()
y_range = (y_min, y_max * 1.1)

# Create the main figure
p = figure(height=300, sizing_mode="stretch_width", tools="", toolbar_location=None, 
           x_axis_type="datetime", x_axis_location="above",
           background_fill_color="#efefef", x_range=(df_bokeh['date'].iloc[-30], df_bokeh['date'].iloc[-1]))
p.vbar(x='date', top='kWh', source=source, width=bar_width, color="seagreen")
p.yaxis.axis_label = 'Daily kWh Usage'
p.y_range = Range1d(*y_range)
# Create the range selection tool figure
select = figure(title="Select Time Range", height=130, sizing_mode="stretch_width", y_range=Range1d(*y_range), 
                x_axis_type="datetime", y_axis_type=None, tools="")
range_tool = RangeTool(x_range=p.x_range)
range_tool.overlay.fill_alpha = 0.2
select.vbar(x='date', top='kWh', source=source, width=bar_width, color="seagreen")
select.ygrid.grid_line_color = None
select.add_tools(range_tool)
p.add_tools(range_tool)
layout = column(p, select, sizing_mode="stretch_width")



from bokeh.embed import json_item
import json
layout = column(p, select, sizing_mode="stretch_width")
json_data = json_item(layout)
with open("energy_usage_plot.json", "w") as f:
    json.dump(json_data, f)
'''
#############################
############## Check for Stationarity ############################
#############################
'''
# show(layout)

# # Rolling stats
# plt.figure(figsize=(14, 6))
# plt.plot(df['kWh'], label="kWh Usage", alpha=0.5)
# plt.plot(df['kWh'].rolling(window=30*48).mean(), label="Rolling Mean (30 days)", color='red')
# plt.plot(df['kWh'].rolling(window=30*48).std(), label="Rolling Std Dev (30 days)", color='blue')
# plt.title("Rolling Mean & Standard Deviation")
# plt.legend()
# plt.show()

# # ADF Test
# def adf_test(series):
#     result = adfuller(series)
#     print(f"ADF Statistic: {result[0]:.4f}")
#     print(f"p-value: {result[1]:.4f}")
#     print(f"Lags Used: {result[2]}")
#     print(f"Observations Used: {result[3]}")
#     print("Stationary" if result[1] <= 0.05 else "Non-stationary")

# adf_test(df['kWh'].dropna())


# '''
# #############################
# ############ time series decomposition ############################
# #############################
# '''

# # 30 min interval
# df['kWh'] = df['kWh'].fillna(method='ffill')
# decomposition = seasonal_decompose(df['kWh'], model='additive', period=30*48)
# fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
# axes[0].plot(df['kWh'], label='Original', color='seagreen')
# axes[0].legend()
# axes[1].plot(decomposition.trend, label='Trend', color='red')
# axes[1].legend()
# axes[2].plot(decomposition.seasonal, label='Seasonality', color='blue')
# axes[2].legend()
# axes[3].plot(decomposition.resid, label='Residuals', color='gray')
# axes[3].legend()
# plt.tight_layout()
# plt.show()


# # daily
# decomposition = seasonal_decompose(df_daily['kWh'], model='additive', period=30) 
# fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
# axes[0].plot(df_daily['kWh'], label='Original', color='seagreen')
# axes[0].legend()
# axes[1].plot(decomposition.trend, label='Trend', color='red')
# axes[1].legend()
# axes[2].plot(decomposition.seasonal, label='Seasonality', color='blue')
# axes[2].legend()
# axes[3].plot(decomposition.resid, label='Residuals', color='gray')
# axes[3].legend()
# plt.tight_layout()
# plt.show()


# '''
# #############################
# ############## autocorrelation and partial autocorrelation
# #############################
# '''

# # hourly
# #auto
# plot_acf(df['kWh'].dropna(), lags=48*30)
# plt.title("Autocorrelation Function (ACF)")
# plt.show()
# # #partial
# # plot_pacf(df['kWh'].dropna(), lags=48*30)
# # plt.title("Partial Autocorrelation Function (PACF)")
# # plt.show()


# # daily
# #auto
# plot_acf(df_daily['kWh'].dropna(), lags=30)
# plt.title("Autocorrelation Function (ACF)")
# plt.show()
# # #partial
# # plot_pacf(df['kWh'].dropna(), lags=48*30)
# # plt.title("Partial Autocorrelation Function (PACF)")
# # plt.show()


# # weekly
# #auto
# plot_acf(df_weekly['kWh'].dropna(), lags=52)
# plt.title("Autocorrelation Function (ACF)")
# plt.show()


# '''
# ##########################################################################################
# ####################### ARIMA MODEL #############################################
# ##########################################################################################
# '''

# '''
# #############################
# ############## daily data
# #############################
# '''
# # Train-Test Split
# train_size = len(df_daily) - 30  # Keep last 30 days for testing
# train_data = df_daily.iloc[:train_size]
# test_data = df_daily.iloc[train_size:]

# # Auto determine ARIMA order
# auto_model = auto_arima(train_data['kWh'], seasonal=False, trace=True, stepwise=True)
# order = auto_model.order

# # Fit ARIMA
# model_arima1 = ARIMA(train_data['kWh'], order=order)
# fitted_model_a1 = model_arima1.fit()

# print(fitted_model_a1.summary())

# '''
# #############################
# ############## 30 min interval model 
# #############################
# '''
# # Train-Test Split
# train_size2 = len(df) - 30*48  # Keep last 30 days for testing
# train_data2 = df_daily.iloc[:train_size2]
# test_data2 = df_daily.iloc[train_size2:]

# # Auto determine ARIMA order
# auto_model_30t = auto_arima(train_data2['kWh'], seasonal=False, trace=True, stepwise=True)
# order = auto_model_30t.order

# # Fit ARIMA
# model_arima2 = ARIMA(train_data2['kWh'], order=order)
# fitted_model_a2 = model_arima2.fit()

# print(fitted_model_a2.summary())
# # model is not as good, I will be using the daily model for forecasting 


# '''
# #############################
# ############### testing output of model ##################
# #############################
# '''

# ######## check residuals for autocorrelation

# residuals_a1 = fitted_model_a1.resid

# print(f"Residuals mean: {residuals_a1.mean()}")
# print(f"Residuals std: {residuals_a1.std()}")
# print(f"Residuals head: {residuals_a1.head()}")

# # ACF plot
# plot_acf(residuals_a1, lags=30)  
# plt.title("Autocorrelation of Residuals")
# plt.show()
# # plt.close()

# # Ljung-Box Test
# ljung_box_test = acorr_ljungbox(residuals_a1, lags=30)
# print(f"Ljung-Box test p-values: {ljung_box_test['lb_pvalue']}")
# """
# The p-values from the Ljung-Box test suggest that the residuals do not exhibit significant autocorrelation at most lags.
# This is generally a good sign because it implies that the model has captured most of the temporal structure in the data
# """


# '''
# #############################
# ################ checking resuidual distribution 
# #############################
# '''

# # qq plot
# stats.probplot(residuals_a1, dist="norm", plot=plt)
# plt.title('Q-Q Plot for Residuals')
# plt.show()
# # plt.close()


# # Histogram for residuals
# sns.histplot(residuals_a1, kde=True)
# plt.title('Histogram of Residuals')
# plt.show()
# # plt.close()


# '''
# ##########################################################################################
# ####################### SARIMA MODEL #############################################
# ##########################################################################################
# '''

# # model_sarima = SARIMAX(df['kWh'], 
# #                 order=(1, 1, 1),  # ARIMA (p, d, q)
# #                 seasonal_order=(1, 1, 1, 365),  # Seasonal ARIMA (P, D, Q, s)
# #                 enforce_stationarity=False, 
# #                 enforce_invertibility=False)

# # # Fit the model
# # fitted_model_sarima = model_sarima.fit(disp=False)

# # # Print model summary
# # print(fitted_model_sarima.summary())

# # print('============================================')

# # model_sarima = SARIMAX(df_daily['kWh'], 
# #                 order=(1, 1, 1),  # ARIMA (p, d, q)
# #                 seasonal_order=(1, 1, 1, 90),  # Seasonal ARIMA (P, D, Q, s)
# #                 enforce_stationarity=False, 
# #                 enforce_invertibility=False)

# # # Fit the model
# # fitted_model_sarima = model_sarima.fit(disp=False)

# # # Print model summary
# # print(fitted_model_sarima.summary())
# # not enough data I will use the ARMIA model

# """
# Not enough data for SARIMA yearly model, would like to have 2/3 cycles and I have just over a year

# """

# '''
# ############################################################
# ############### forecasting with ARIMA ########################
# ############################################################
# '''
# '''
# #############################
# ################ basic forecast
# #############################
# '''
# # Forecast Next 30 Days
# forecast = fitted_model_a1.forecast(steps=30)

# # Compare Forecast vs Actual
# actual = df_daily['kWh'].iloc[-30:].values 
# mae = mean_absolute_error(actual, forecast)
# rmse = np.sqrt(mean_squared_error(actual, forecast))
# mape = np.mean(np.abs((actual - forecast) / actual)) * 100
# print(f"Mean Absolute Error (MAE): {mae:.4f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
# print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")

# # summing per month to see difference
# total_forecast = round(forecast.sum(), 2)
# total_actual = actual.sum()
# print(f"Total Forecasted Usage for the month: {total_forecast} kWh")
# print(f"Total Actual Usage for the month: {total_actual} kWh")

# # Plot Results
# plt.figure(figsize=(12, 5))
# plt.plot(df_daily.index, df_daily['kWh'], label="Actual")
# plt.plot(df_daily.index[-30:], forecast, label="Forecast", linestyle='dashed', color='red')
# plt.legend()
# plt.title("ARIMA Forecast vs Actual Data")
# plt.show()
# # plt.close()



# '''
# #############################
# ############### rolling forecast
# #############################
# '''
# train, test = df_daily['kWh'][:-30], df_daily['kWh'][-30:]
# predictions = []  # Store forecasts
# forecasted_values = []
# # Rolling forecast loop
# for t in range(len(test)):
#     model = ARIMA(train, order=(1,1,1), enforce_invertibility=False, enforce_stationarity=False) 
#     model_fit = model.fit()
    
#     forecast = model_fit.forecast(steps=1)
#     forecasted_values.append(float(forecast.item()))  

#     predictions.append(float(forecast.item()))  

#     train = np.append(train, test.iloc[t].item()) 

# # Evaluate performance
# mae = mean_absolute_error(test, predictions)
# rmse = np.sqrt(mean_squared_error(test, predictions))
# mape = np.mean(np.abs((test - predictions) / test)) * 100
# print(f"Rolling Forecast MAE: {mae:.4f}")
# print(f"Rolling Forecast RMSE: {rmse:.4f}")
# print(f"Rolling Forecast MAPE: {mape:.4f}")

# # sum for month to see difference in prediction 
# total_forecast_rolling = round(sum(forecasted_values), 2)
# total_actual_rolling = test.sum()
# print(f"Total Rolling Forecasted Usage for the month: {total_forecast_rolling} kWh")
# print(f"Total Actual Usage for the month: {total_actual_rolling} kWh")

# # plot results for visual
# plt.figure(figsize=(12, 5))
# plt.plot(df_daily.index, df_daily['kWh'], label="Actual")
# plt.plot(df_daily.index[-len(forecasted_values):], forecasted_values, label="Forecast", linestyle='dashed', color='red')
# plt.ylabel("kWh")
# plt.legend()
# plt.title("ARIMA Rolling Forecast vs Actual Data")
# plt.show()
# # plt.close()
