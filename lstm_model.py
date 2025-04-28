import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import holidays
from datetime import datetime, timedelta
from meteostat import Hourly, Point
from sklearn.model_selection import train_test_split



'''
#############################
############ READ IN DATA, sort and merge ######################
#############################
'''
ml_data_1 = pd.read_csv("./Cleaned_Data/modified_cleaned_home_energy_usage.csv", parse_dates=['Timestamp'])
ml_data_2 = pd.read_csv("./Cleaned_Data/modified_cleaned_home_energy_usage2.csv", parse_dates=['Timestamp'])
ml_data = pd.concat([ml_data_1, ml_data_2], ignore_index=True) 
ml_data = ml_data.drop_duplicates(subset=['Timestamp'])
# remove dates in april 2025
aprilData = ml_data[(ml_data['Timestamp'].dt.year == 2025) & (ml_data['Timestamp'].dt.month == 4)]
ml_data = ml_data[~((ml_data['Timestamp'].dt.year == 2025) & (ml_data['Timestamp'].dt.month == 4))]

# ml_data.to_csv("./Cleaned_Data/combined_home_energy_usage5.csv", index=False)
ml_data = ml_data.sort_values("Timestamp")
# data is just from 2024-02-10 to 2024-03-31

# weather data 
location = Point(38.8799, -77.1068) # arlington VA
start = datetime(2024, 2, 10)
end = datetime(2025, 3, 31)  # Extend this as needed #=======================================

# get weather data and interpolate to 30 min intervals
weather = Hourly(location, start, end).fetch()
weather = weather.resample('30T').interpolate()
weather = weather.reset_index()
weather = weather[['time', 'temp', 'dwpt', 'rhum', 'wspd']]
weather = weather.rename(columns={'time': 'Timestamp'})

# Merge weather data with electricity data
ml_data = pd.merge(ml_data, weather, on='Timestamp', how='inner')
# ml_data.to_csv("./Cleaned_Data/combined_home_energy_usage7.csv", index=False)

# checking for null values
# print(ml_data.isna().sum())
# print(ml_data.dtypes)
# check full timeline to see if there are any missing timestamps
# full_range = pd.date_range(start=ml_data['Timestamp'].min(), end=ml_data['Timestamp'].max(), freq='H')
# missing_times = full_range.difference(ml_data['Timestamp'])
# print("Missing timestamps:")
# print(missing_times)

ml_data = ml_data.dropna(subset=['kWh'])



'''
#############################
############ features ######################
#############################
'''
# weekend
ml_data['is_weekend'] = ml_data['day_of_week'].apply(lambda x: 1 if x >= 4 else 0)

# holidays
us_holidays = holidays.US(years=[2024, 2025])
holidays_list = set(us_holidays.keys())
ml_data['is_holiday'] = ml_data['Timestamp'].dt.date.apply(lambda date: 1 if date in us_holidays else 0)

# hour of the week
ml_data['hour_of_week'] = ml_data['day_of_week'] * 24 + ml_data['hour']

# Lagged kWh (1 lag)
ml_data['lag1_kWh'] = ml_data['kWh'].shift(1)
ml_data = ml_data.dropna(subset=['lag1_kWh'])



# feature list
features = ['kWh', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
            'hour_of_week', 'lag1_kWh', 'temp', 'dwpt', 'rhum', 'wspd']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(ml_data[features])



'''
#############################
############ Create sequences for LSTM 
#############################
'''
def create_sequences(data, seq_length=336):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])  # kWh is the target (index 0)
    return np.array(X), np.array(y)

SEQ_LEN = 336  # 7 days of 30 min data
X, y = create_sequences(scaled_data, seq_length=SEQ_LEN)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

'''
#############################
############ build model
#############################
'''
model = Sequential([
    LSTM(64, activation='tanh', return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(32, activation='tanh'),
    Dropout(0.2),
    Dense(1)
])

# Huber loss
model.compile(optimizer='adam', loss=Huber(delta=1.0))

# earlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fit the model
model.fit(
    X_train, y_train,
    epochs=50, 
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)


######
# check for over fitting
#######
history = model.fit(
    X_train, y_train,
    epochs=50, 
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()






'''
#############################
############ Forecast the next 30 days hourly
#############################
'''

# function for dummy data 
def build_forecast_dummy(pred_kwh, lag1_kwh, future_time, weather_features, scaler, features_list, holidays_list):
    dummy = np.zeros((1, len(features_list)))

    # Fill features
    dummy[0, 0] = pred_kwh
    dummy[0, 1] = future_time.hour / 23.0      # hour (normalized)
    dummy[0, 2] = future_time.weekday() / 6.0  # day_of_week (normalized)
    dummy[0, 3] = future_time.month / 12.0     # month (normalized)
    dummy[0, 4] = 1.0 if future_time.date() in holidays_list else 0.0  # is_holiday
    dummy[0, 5] = 1.0 if future_time.weekday() >= 4 else 0.0           # is_weekend # changing to 4 to include friday
    dummy[0, 6] = (future_time.weekday() * 24 + future_time.hour) / 167.0  # hour_of_week
    dummy[0, 7] = lag1_kwh  # previous kWh prediction (lag1)
    dummy[0, 8:] = weather_features  # temp, dewpoint, humidity, windspeed

    dummy_df = pd.DataFrame(dummy, columns=features_list)
    return scaler.transform(dummy_df)[0]



last_year_month = ml_data[(ml_data['Timestamp'].dt.month == 4) & (ml_data['Timestamp'].dt.year == 2024)]
weather_by_hour = last_year_month.groupby('hour')[['temp', 'dwpt', 'rhum', 'wspd']].mean().reset_index()
weather_by_hour.columns = ['hour', 'temp', 'dwpt', 'rhum', 'wspd']
weather_by_hour.reset_index(inplace=True)

forecast_hours = 24 * 30
predictions = []
current_input = X[-1]
last_timestamp = ml_data['Timestamp'].iloc[-1]
last_lag1_kwh_scaled = current_input[-1][0]

for i in range(forecast_hours):
    future_time = last_timestamp + timedelta(hours=i+1)
    hour = future_time.hour

    # Lookup weather
    w = weather_by_hour.loc[weather_by_hour['hour'] == hour, ['temp', 'dwpt', 'rhum', 'wspd']].iloc[0].values

    # Predict kWh (use last prediction if available)
    last_kwh_scaled = predictions[-1] if predictions else current_input[-1][0]

    # Build full input vector
    new_input_scaled = build_forecast_dummy(
        pred_kwh=last_kwh_scaled,
        lag1_kwh=last_lag1_kwh_scaled,
        future_time=future_time,
        weather_features=w,
        scaler=scaler,
        features_list=features,
        holidays_list=holidays_list
    )

    # Add to sequence
    current_input = np.vstack((current_input[1:], new_input_scaled))

    # Predict next value
    pred = model.predict(current_input.reshape(1, *current_input.shape), verbose=0)[0][0]
    predictions.append(pred)

    # Update lag1_kwh for next iteration
    last_lag1_kwh_scaled = last_kwh_scaled

'''
#############################
############ Inverse transform kWh predictions
#############################
'''
pred_scaled = np.array(predictions).reshape(-1, 1)
dummy_input = np.zeros((len(predictions), len(features)))
dummy_input[:, 0] = pred_scaled[:, 0]
inv_preds = scaler.inverse_transform(dummy_input)[:, 0]


'''
#############################
############ plot
#############################
'''
# plt.figure(figsize=(15, 5))
# plt.plot(inv_preds)
# plt.title("Predicted Electricity Usage for Next 30 Days (Hourly)")
# plt.xlabel("Hour")
# plt.ylabel("kWh")
# plt.grid(True)
# plt.tight_layout()
# plt.show()



'''
#############################
############ plot 2
#############################
'''
# time index for the prediction (April 2025)
forecast_hours = len(inv_preds)
time_index_future = pd.date_range(start=datetime(2025, 4, 1), periods=forecast_hours, freq='h')

# plotting both
plt.figure(figsize=(15, 5))

# Forecasted data (red)
plt.plot(time_index_future, inv_preds, color='red', label='April 2025 (Predicted)')

# actual april data (green)
plt.plot(aprilData['Timestamp'], aprilData['kWh'], color='green', label='April 2025 (Actual)')

# Labels and formatting
plt.title("Electricity Usage: March (Actual) vs April (Forecast)")
plt.xlabel("Date and Time")
plt.ylabel("kWh")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()