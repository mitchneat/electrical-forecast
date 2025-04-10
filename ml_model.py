# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt 
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.preprocessing import MinMaxScaler


# '''
# #############################
# ############ READ IN DATA and group by day ######################
# #############################
# '''
# df = pd.read_csv("./Cleaned_Data/cleaned_home_energy_usage.csv")
# df['Timestamp'] = pd.to_datetime(df['Timestamp'])
# df = df.set_index("Timestamp")


# # Scale data
# scaler = MinMaxScaler()
# df_scaled = scaler.fit_transform(df['kWh'].values.reshape(-1, 1))

# # Prepare sequences
# X, y = [], []
# window_size = 48 * 7  # 7-day lookback
# for i in range(len(df_scaled) - window_size):
#     X.append(df_scaled[i:i+window_size])
#     y.append(df_scaled[i+window_size])

# X, y = np.array(X), np.array(y)

# # Train-test split
# train_size = int(len(X) * 0.8)
# X_train, y_train = X[:train_size], y[:train_size]
# X_test, y_test = X[train_size:], y[train_size:]

# # Build LSTM model
# model = Sequential([
#     LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)),
#     LSTM(50, activation='relu'),
#     Dense(1)
# ])

# model.compile(optimizer='adam', loss='mse')
# model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# # Predictions
# y_pred = model.predict(X_test)
# y_pred = scaler.inverse_transform(y_pred)

# # Plot results
# plt.plot(df.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), label="Actual")
# plt.plot(df.index[-len(y_test):], y_pred, label="LSTM Prediction")
# plt.legend()
# plt.show()
