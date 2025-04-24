import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime
from meteostat import Hourly, Point
from sklearn.model_selection import train_test_split

'''
#############################
############ READ IN DATA, sort and merge ######################
#############################
'''
ml_data = pd.read_csv("./Cleaned_Data/modified_cleaned_home_energy_usage.csv", parse_dates=['Timestamp'])
ml_data = ml_data.sort_values("Timestamp")

# weather data 
location = Point(38.8799, -77.1068) # arlington VA
start = datetime(2024, 2, 10)
end = datetime(2024, 3, 10)  # Extend this as needed

weather = Hourly(location, start, end).fetch()
weather = weather.reset_index()
weather = weather[['time', 'temp', 'dwpt', 'rhum', 'wspd']]
weather = weather.rename(columns={'time': 'Timestamp'})

# Merge weather data with electricity data
ml_data = pd.merge(ml_data, weather, on='Timestamp', how='inner')

# checking for null values
# print(ml_data.isna().sum())
# print(ml_data.dtypes)


'''
#############################
############ Normalize all features (including weather) ######################
#############################
'''
features = ['kWh', 'hour', 'day_of_week', 'month', 'temp', 'dwpt', 'rhum', 'wspd']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(ml_data[features])



'''
#############################
############ Create sequences for LSTM 
#############################
'''
def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])  # kWh is the target (index 0)
    return np.array(X), np.array(y)

SEQ_LEN = 24
X, y = create_sequences(scaled_data, seq_length=SEQ_LEN)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

'''
#############################
############ build model
#############################
'''
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

'''
#############################
############ Forecast the next 30 days hourly
#############################
'''
forecast_hours = 24 * 30
predictions = []
current_input = X[-1]

for _ in range(forecast_hours):
    pred = model.predict(current_input.reshape(1, *current_input.shape), verbose=0)[0][0]
    predictions.append(pred)

    # Use last known non-kWh features
    last_features = current_input[-1][1:]
    new_input = np.hstack(([pred], last_features))
    current_input = np.vstack((current_input[1:], new_input))


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
plt.figure(figsize=(12, 4))
plt.plot(inv_preds)
plt.title("Predicted Electricity Usage for Next 30 Days (Hourly)")
plt.xlabel("Hour")
plt.ylabel("kWh")
plt.grid(True)
plt.tight_layout()
plt.show()