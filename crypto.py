import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# -----------------
# Page Configuration
# -----------------
st.set_page_config(
    page_title="Crypto Price Predictor",
    page_icon="🪙",
    layout="wide"
)

st.title("🪙 Systematic Crypto Price Predictor")
st.write("Analyze and forecast cryptocurrency prices using time series models.")

# -----------------
# Sidebar Inputs
# -----------------
st.sidebar.header("User Inputs")
# List of popular cryptos
cryptos = ('BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'SHIB-USD', 'ADA-USD')
selected_crypto = st.sidebar.selectbox('Select Cryptocurrency', cryptos)

# Date range
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('today'))

# Forecast period
n_days = st.sidebar.slider('Days to Forecast', 1, 365, 30)

# -----------------
# Data Loading
# -----------------
# Use Streamlit's cache to avoid re-downloading data
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, auto_adjust=True)
        if data.empty:
            st.error("No data found for the selected ticker and date range.")
            return None
        data.reset_index(inplace=True)
        
        # Handle multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Ensure Date column is datetime
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data_load_state = st.info(f'Loading data for {selected_crypto}...')
data = load_data(selected_crypto, start_date, end_date)
data_load_state.empty() # Clear the 'loading' message

if data is not None:
    # Debug: Show data info
    st.sidebar.write(f"Data shape: {data.shape}")
    st.sidebar.write(f"Columns: {list(data.columns)}")
    st.sidebar.write(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    st.sidebar.write(f"Sample data:")
    st.sidebar.dataframe(data.head())
    
    # -----------------
    # Main App Logic
    # -----------------
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Exploratory Data Analysis", "Forecasting with Linear Regression", "Model Evaluation"])

    # -----------------
    # Tab 1: EDA
    # -----------------
    with tab1:
        st.subheader(f"Raw Data for {selected_crypto}")
        st.dataframe(data.tail())

        # Plotly Candlestick Chart
        st.subheader("Price Candlestick Chart")
        fig_candle = go.Figure()
        fig_candle.add_trace(go.Candlestick(
            x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Market Data'
        ))
        fig_candle.update_layout(
            title=f"{selected_crypto} Price Action", 
            xaxis_rangeslider_visible=True,
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_candle, use_container_width=True)

        # Simple Moving Average
        st.subheader("Price with Moving Averages (50 & 200 days)")
        data['MA50'] = data['Close'].rolling(50).mean()
        data['MA200'] = data['Close'].rolling(200).mean()
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(
            x=data['Date'], 
            y=data['Close'], 
            name='Close Price',
            line=dict(color='blue', width=2)
        ))
        fig_ma.add_trace(go.Scatter(
            x=data['Date'], 
            y=data['MA50'], 
            name='50-Day MA',
            line=dict(color='orange', width=2)
        ))
        fig_ma.add_trace(go.Scatter(
            x=data['Date'], 
            y=data['MA200'], 
            name='200-Day MA',
            line=dict(color='red', width=2)
        ))
        fig_ma.update_layout(
            title="Price with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_ma, use_container_width=True)

    # -----------------
    # Tab 2: Price Forecasting
    # -----------------
    with tab2:
        st.subheader("Price Forecasting with Linear Regression")
        
        # Prepare data
        df_train = data[['Date', 'Close']].copy()
        df_train['Days'] = (df_train['Date'] - df_train['Date'].min()).dt.days
        
        # Train model
        X = df_train[['Days']].values
        y = df_train['Close'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Create forecast
        last_day = df_train['Days'].max()
        future_days = np.arange(last_day + 1, last_day + n_days + 1).reshape(-1, 1)
        future_predictions = model.predict(future_days)
        
        # Create forecast dataframe
        forecast_dates = pd.date_range(start=df_train['Date'].max() + pd.Timedelta(days=1), periods=n_days)
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted_Price': future_predictions.flatten()
        })
        
        # Display forecast data
        st.subheader('Forecast Data')
        st.dataframe(forecast_df)

        # Plot forecast
        st.subheader(f'Forecast Plot for {n_days} Days')
        
        # Debug info
        st.write(f"Historical data points: {len(df_train)}")
        st.write(f"Forecast data points: {len(forecast_df)}")
        st.write(f"Historical price range: {float(df_train['Close'].min()):.2f} - {float(df_train['Close'].max()):.2f} USD")
        st.write(f"Forecast price range: {float(forecast_df['Predicted_Price'].min()):.2f} - {float(forecast_df['Predicted_Price'].max()):.2f} USD")
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df_train['Date'], 
            y=df_train['Close'], 
            name='Historical Price', 
            mode='lines',
            line=dict(color='blue', width=2)
        ))
        fig1.add_trace(go.Scatter(
            x=forecast_df['Date'], 
            y=forecast_df['Predicted_Price'], 
            name='Forecast', 
            mode='lines', 
            line=dict(dash='dash', color='red', width=2)
        ))
        fig1.update_layout(
            title=f"Price Forecast for {n_days} Days",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            showlegend=True
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Show trend info
        st.subheader('Trend Information')
        daily_change = model.coef_[0].item()
        trend_direction = "upward" if daily_change > 0 else "downward"
        st.write(f"**Trend:** The model shows an {trend_direction} trend.")
        st.write(f"**Daily Change:** ${daily_change:.2f} per day on average.")
        
    # -----------------
    # Tab 3: Model Evaluation
    # -----------------
    with tab3:
        st.subheader("Model Evaluation")
        
        # Calculate in-sample predictions (how well the model fits historical data)
        predictions = model.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        
        st.write(f"**Metrics on Historical Data (In-Sample Fit):**")
        st.write(f"- **Mean Absolute Error (MAE):** ${mae:.2f}")
        st.write(f"- **Mean Squared Error (MSE):** ${mse:.2f}")
        st.write(f"- **Root Mean Squared Error (RMSE):** ${rmse:.2f}")

        st.info("""
        **Note:** This evaluation is "in-sample" (how well the model fit the data it was trained on). 
        Lower values indicate better fit. MAE represents the average prediction error in dollars.
        """)
        
        # Plot actual vs predicted
        st.subheader("Actual vs Predicted Prices")
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            x=df_train['Date'], 
            y=y, 
            name='Actual Price', 
            mode='lines',
            line=dict(color='blue', width=2)
        ))
        fig_eval.add_trace(go.Scatter(
            x=df_train['Date'], 
            y=predictions, 
            name='Predicted Price', 
            mode='lines', 
            line=dict(dash='dot', color='orange', width=2)
        ))
        fig_eval.update_layout(
            title="Model Fit: Actual vs Predicted Prices",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_eval, use_container_width=True)

else:
    st.warning("Please select a valid cryptocurrency and date range to begin.")
