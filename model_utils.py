import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

def get_stock_data(symbol, start='2015-01-01'):
    df = yf.download(symbol, start=start)
    return df['Close'].dropna()

def prepare_lstm_data(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_next_price(data):
    X, y, scaler = prepare_lstm_data(data)
    model = build_model((X.shape[1], 1))
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    last_60 = data[-60:].values.reshape(-1, 1)
    last_60_scaled = scaler.transform(last_60).reshape(1, 60, 1)
    predicted_scaled = model.predict(last_60_scaled)
    return scaler.inverse_transform(predicted_scaled)[0][0]