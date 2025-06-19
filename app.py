import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model_utils import get_stock_data, predict_next_price

st.set_page_config(page_title="Nifty 500 Stock Predictor", layout="centered")
st.title("📈 Nifty 500 Stock Price Predictor (LSTM)")

df = pd.read_csv("nifty500_list.csv")
df['Symbol'] = df['Symbol'].astype(str).apply(lambda x: x.strip().upper() + '.NS' if not x.endswith('.NS') else x)
symbols = dict(zip(df['Symbol'], df['Company Name']))

selected_symbol = st.selectbox("Choose a company", options=symbols.keys(), format_func=lambda x: symbols[x])

if st.button("Predict Price"):
    with st.spinner("Fetching data and training model..."):
        try:
            data = get_stock_data(selected_symbol)
            prediction = predict_next_price(data)

            st.success(f"Predicted Next Closing Price: ₹{prediction:.2f}")

            st.subheader("Recent Price Trend")
            fig, ax = plt.subplots()
            data[-100:].plot(ax=ax)
            ax.set_title(symbols[selected_symbol])
            ax.set_ylabel("Price (INR)")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {str(e)}")