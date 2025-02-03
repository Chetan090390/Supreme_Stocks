import streamlit as st
import sqlite3
import bcrypt
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

# Database setup
conn = sqlite3.connect('user_credentials.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT PRIMARY KEY, password TEXT)''')

# Authentication functions
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password)

def register_user(username, password):
    hashed_password = hash_password(password)
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    if result:
        return verify_password(result[0], password)
    return False

# Stock prediction functions
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def fetch_data(ticker, period='1y', interval='1d'):
    return yf.download(ticker, period=period, interval=interval)

def plot_predictions(real_prices, predictions, title):
    plt.figure(figsize=(12, 6))
    plt.plot(real_prices, color='blue', label='Actual Prices')
    colors = ['red', 'green', 'orange']
    labels = ['LSTM', 'Linear Regression', 'Decision Tree']
    for i, pred in enumerate(predictions):
        plt.plot(range(len(real_prices) - len(pred), len(real_prices)), pred, color=colors[i], label=f'{labels[i]} Predictions')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price USD ($)')
    plt.legend()
    st.pyplot(plt)

def display_metric(label, value, trend):
    color = "green" if trend == "Upward" else "red"
    arrow = "↑" if trend == "Upward" else "↓"
    st.markdown(f"""
        <div style="background-color: #222; padding: 10px; border-radius: 5px;">
            <h3 style="color:white;">{label}</h3>
            <h1 style="color:white;">${value:.2f}</h1>
            <h2 style="color:{color};">{arrow} {trend}</h2>
        </div>
        """, unsafe_allow_html=True)

def stock_prediction_app():
    st.title('Stock Price Prediction App')

    ticker = st.text_input('Enter Stock Ticker:', 'AAPL')
    prediction_type = st.selectbox('Choose Prediction Type:', ['Real-Time', 'Historical'])

    if st.button('Fetch Data and Predict'):
        with st.spinner('Fetching data...'):
            if prediction_type == 'Historical':
                data = fetch_data(ticker)
            else:
                data = fetch_data(ticker, period='1d', interval='1m')

            if data.empty:
                st.write("The stock ticker you entered is not valid or there is no data available.")
                return

            st.write(f"Displaying data for {ticker}:")
            st.line_chart(data['Close'])

            # Preprocess the data
            data = data[['Close']]
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_data = scaler.fit_transform(data)

            # Prepare data for predictions
            seq_length = 60
            X, y = create_sequences(scaled_data, seq_length)

            # Train models
            X_train = np.reshape(X, (X.shape[0], X.shape[1], 1))

            # LSTM Model
            model_lstm = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=25),
                Dense(units=1)
            ])
            model_lstm.compile(optimizer='adam', loss='mean_squared_error')
            model_lstm.fit(X_train, y, epochs=5, batch_size=32, verbose=0)
            lstm_predictions = model_lstm.predict(X_train)
            lstm_predictions = scaler.inverse_transform(lstm_predictions)

            # Linear Regression Model
            model_lr = LinearRegression()
            X_train_lr = np.reshape(X, (X.shape[0], X.shape[1]))
            model_lr.fit(X_train_lr, y)
            lr_predictions = model_lr.predict(X_train_lr)
            lr_predictions = scaler.inverse_transform(lr_predictions.reshape(-1, 1))

            # Decision Tree Model
            model_dt = DecisionTreeRegressor()
            model_dt.fit(X_train_lr, y)
            dt_predictions = model_dt.predict(X_train_lr)
            dt_predictions = scaler.inverse_transform(dt_predictions.reshape(-1, 1))

            # Display predictions and trends
            last_closing_price = data['Close'].values[-1]
            next_day_prediction_lstm = lstm_predictions[-1][0]
            next_day_prediction_lr = lr_predictions[-1][0]
            next_day_prediction_dt = dt_predictions[-1][0]

            trend_lstm = "Upward" if next_day_prediction_lstm > last_closing_price else "Downward"
            trend_lr = "Upward" if next_day_prediction_lr > last_closing_price else "Downward"
            trend_dt = "Upward" if next_day_prediction_dt > last_closing_price else "Downward"

            st.markdown("## Predictions for the Next Period")

            # Display custom styled predictions
            display_metric("Tomorrow's Closing Price by LSTM", next_day_prediction_lstm, trend_lstm)
            display_metric("Tomorrow's Closing Price by Linear Regression", next_day_prediction_lr, trend_lr)
            display_metric("Tomorrow's Closing Price by Decision Tree", next_day_prediction_dt, trend_dt)

            # Plotting Predictions
            plot_predictions(
                data['Close'].values,
                [lstm_predictions, lr_predictions, dt_predictions],
                f'{ticker} Stock Price Predictions ({prediction_type})'
            )

def main():
    st.set_page_config(page_title="Stock Prediction App", layout="wide")

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        stock_prediction_app()
        if st.sidebar.button('Logout'):
            st.session_state.logged_in = False
            st.rerun()
    else:
        st.title("Welcome to GROWMORE ")

        menu = ["Login", "Register"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Login":
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')

            if st.button("Login"):
                if login_user(username, password):
                    st.success(f"Logged in successfully as {username}")
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Incorrect username or password")

        elif choice == "Register":
            st.subheader("Create New Account")
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type='password')
            confirm_password = st.text_input("Confirm Password", type='password')

            if st.button("Register"):
                if new_password == confirm_password:
                    if register_user(new_username, new_password):
                        st.success("Account created successfully")
                    else:
                        st.error("Username already exists")
                else:
                    st.error("Passwords do not match")

if __name__ == '__main__':
    main()

