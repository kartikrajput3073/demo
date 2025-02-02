import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
from datetime import date, timedelta
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# App title and description
app = 'Stock Market Forecaster'
st.title(app)
st.subheader("Forecasting Stock Price of Selected Company")
st.image("https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")

# Sidebar for user input
st.sidebar.header('Select Parameters')

# Date input for start and end dates
sdate = st.sidebar.date_input('Start Date', datetime.date(2014, 1, 1))
edate = st.sidebar.date_input('End Date', date.today())

# List of ticker symbols for user to select from
ticker_list = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOG', 'META', 'TSM', 'NVDA', 'NFLX', 'AMD']
ticker = st.sidebar.selectbox('Select Company', ticker_list)

# Download data using yfinance
data = yf.download(ticker, start=sdate, end=edate)
data.insert(0, 'Date', data.index, True)
data.reset_index(drop=True, inplace=True)
st.write('Data from', sdate, 'to', edate)
st.write(data)

# Data visualization
st.header('Data Visualization')
st.subheader('Plot of the Data')
fig = px.line(data, x='Date', y=data.columns, title='Stock Price of Selected Company', template='plotly_dark', width=950, height=600)
st.plotly_chart(fig)

# Select column to forecast
column = st.selectbox('Select the Column to Forecast', data.columns[1:])
data = data[['Date', column]]
st.write('Selected Column')
st.write(data)

# Check if data is stationary using ADF test
st.header('Is Data Stationary?')
st.write(adfuller(data[column])[1] < 0.05)

# Decompose the time series data
st.header('Decomposed Data')
decomposition = seasonal_decompose(data[column], model='additive', period=30)
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title="Trend", width=950, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='blue'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title="Seasonality", width=950, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='green'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title="Residuals", width=950, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='red', line_dash='dot'))

# User input for SARIMAX parameters
p = st.number_input('Select the value of p [previous days prices to look at]', value=2)
q = st.number_input('Select the value of q [difference the data to make it stable]', value=2)
d = st.number_input('Select the value of d [past prediction errors to consider]', value=2)
seasonal_order = st.number_input('Select the value of seasonal p', value=12)

# Build and fit SARIMAX model
model = sm.tsa.statespace.SARIMAX(data[column], order=(p, d, q), seasonal_order=(p, d, q, 12))
model = model.fit()

# Display model summary
#st.header('Model Summary')
#st.write(model.summary())
#st.write("---")

# Forecasting
st.write(" ")
st.write("<p style='color:#336fbb; font-size:50px; font-weight:bold;'>Forecasting the Data</p>", unsafe_allow_html=True)
forecast_period = st.number_input('Enter the number of days to forecast', 1, 365, 30)
predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period)
predictions = predictions.predicted_mean

# Prepare predictions DataFrame
predictions.index = pd.date_range(start=edate, periods=len(predictions), freq='D')
predictions = pd.DataFrame(predictions)

predictions.insert(0, 'Date', predictions.index, True)
predictions.reset_index(drop=True, inplace=True)

# Display actual and predicted data
st.write("## Predicted Data", predictions)
st.write("## Actual Data", data)
st.write("---")

# Plot actual vs predicted data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions["predicted_mean"], mode='lines', name='Predicted', line=dict(color='red')))
fig.update_layout(title='Actual vs. Predicted', xaxis_title='Date', yaxis_title='Price', width=950, height=600)
st.plotly_chart(fig)

# Option to show separate plots
show_plots = False
if st.button('Show separate plots'):
    if not show_plots:
        st.write(px.line(x=data["Date"], y=data[column], title="Actual", width=950, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='blue'))
        st.write(px.line(x=predictions["Date"], y=predictions["predicted_mean"], title="Predicted", width=950, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='red'))
    show_plots = True
else:
    show_plots = False

st.write("---")

# Social media links
st.subheader("Connect with me on Social Media")

insta_logo = "https://icones.pro/wp-content/uploads/2021/02/instagram-logo-icone4.png"
insta_url = "https://www.instagram.com/lowkey_kartik?utm_source=qr&igsh=dTg5ZmZmMzlma2Yx"
github_logo = "https://i.pinimg.com/736x/b5/1b/78/b51b78ecc9e5711274931774e433b5e6.jpg"
github_url = "https://github.com/kartikrajput3073/demo"
st.markdown(f'<a href="{insta_url}"><img src="{insta_logo}" width="60" height="60"></a>', unsafe_allow_html=True)
st.markdown(f'<a href="{github_url}"><img src="{github_logo}" width="60" height="60"></a>', unsafe_allow_html=True)
