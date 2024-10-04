import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Define start and end dates
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Page title
st.title('Stock Forecast App')

# Define stocks and input parameters
stocks = (
    "AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "FB", "NVDA", "JPM", "JNJ", "V",
    "PG", "HD", "DIS", "MA", "PYPL", "NFLX", "INTC", "CMCSA", "ADBE", "VZ",
    "KO", "PEP", "MRK", "PFE", "XOM", "T", "CSCO", "WMT", "BA", "UNH", "COST",
    "MCD", "IBM", "GS", "CAT", "NKE", "BLK", "GE", "F", "BTU"
)
selected_stock = st.selectbox('Select dataset for prediction', stocks)
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Load stock data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Train Prophet model and forecast
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast data
st.subheader('Forecast data')
st.write(forecast.tail())
st.write(f'Forecast plot for {n_years} years')
st.plotly_chart(plot_plotly(m, forecast))
st.write("Forecast components")
st.write(m.plot_components(forecast))

# Back button
if st.button("Back"):
    st.experimental_set_query_params(page="main")  # Redirect to the main page or a specified one
    st.experimental_rerun()
