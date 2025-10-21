# 🪙 Crypto Price Predictor

A comprehensive cryptocurrency price analysis and forecasting web application built with Streamlit, yfinance, and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

### 📈 Interactive Data Visualization
- **Candlestick Charts**: Professional-grade price action visualization with OHLC (Open, High, Low, Close) data
- **Moving Averages**: 50-day and 200-day moving averages for trend analysis
- **Interactive Plotly Charts**: Zoom, pan, and hover for detailed data exploration

### 🤖 Price Forecasting
- **Linear Regression Model**: Machine learning-based price prediction
- **Customizable Forecast Period**: Predict prices from 1 to 365 days into the future
- **Trend Analysis**: Automatic detection of upward/downward trends with daily price change metrics

### ℹ️ Model Evaluation
- **Performance Metrics**: 
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
- **Visual Comparison**: Actual vs Predicted price charts for model validation

### 💎 Supported Cryptocurrencies
- Bitcoin (BTC-USD)
- Ethereum (ETH-USD)
- Solana (SOL-USD)
- Dogecoin (DOGE-USD)
- Shiba Inu (SHIB-USD)
- Cardano (ADA-USD)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/studentofstars/Crypto-predictor.git
cd Crypto-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run crypto.py
```

4. **Open your browser**
The app will automatically open at `http://localhost:8501`

## 📊 How to Use

1. **Select Cryptocurrency**: Choose from the dropdown in the sidebar
2. **Set Date Range**: Pick your desired start and end dates for historical data
3. **Choose Forecast Period**: Use the slider to select how many days to forecast (1-365)
4. **Explore the Tabs**:
   - **Tab 1 - EDA**: View raw data and interactive charts
   - **Tab 2 - Forecasting**: See price predictions and trend information
   - **Tab 3 - Model Evaluation**: Review model performance metrics

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Data Source**: yfinance (Yahoo Finance API)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Machine Learning**: Scikit-learn (Linear Regression)

## 📁 Project Structure

```
crypto-price-predictor/
│
├── crypto.py              # Main application file
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 🔧 Configuration

### Customizing Date Range
Default settings in the sidebar:
- **Start Date**: January 1, 2020
- **End Date**: Today's date
- **Forecast Days**: 30 (adjustable 1-365)

### Adding More Cryptocurrencies
Edit the `cryptos` tuple in `crypto.py`:
```python
cryptos = ('BTC-USD', 'ETH-USD', 'YOUR-CRYPTO-USD')
```

## 📈 Machine Learning Model

The application uses **Linear Regression** for time series forecasting:

1. **Feature Engineering**: Converts dates to numerical "days since start"
2. **Training**: Fits a linear model on historical price data
3. **Prediction**: Extrapolates future prices based on the learned trend
4. **Evaluation**: Calculates error metrics on the training data

### Model Limitations
- Linear regression assumes a linear trend, which may not capture complex market dynamics
- Best suited for short to medium-term forecasts
- Does not account for external market factors or sudden events

## 🤝 Contributing

Contributions are welcome! Here are some ways you can contribute:

1. **Report Bugs**: Open an issue describing the bug
2. **Suggest Features**: Propose new features or improvements
3. **Submit Pull Requests**: Fork the repo and create a pull request

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/studentofstars/Crypto-predictor.git
cd Crypto-predictor

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run crypto.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This application is for educational and informational purposes only.**

- Cryptocurrency investments are highly volatile and risky
- Past performance does not guarantee future results
- This tool should NOT be used as the sole basis for investment decisions
- Always do your own research and consult with financial advisors
- The developers are not responsible for any financial losses

## 🙏 Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) - For providing free financial data
- [Streamlit](https://streamlit.io/) - For the amazing web framework
- [Plotly](https://plotly.com/) - For interactive visualization capabilities

## 📧 Contact

For questions, suggestions, or feedback, please open an issue on GitHub.

---

**Built with ❤️ using Python and Streamlit**

*Star ⭐ this repo if you find it helpful!*
