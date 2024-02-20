
import streamlit as st
import yfinance  as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import datetime
from datetime import date, timedelta
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

app='Stock Market Forecaster'

st.title(app)
st.subheader("Forecasting Stock Price of Selected Company")
st.image("https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
st.sidebar.header('Select Parameters')

sdate=st.sidebar.date_input('Start Date',datetime.date(2010,1,1))
edate=st.sidebar.date_input('End Date',date.today())

ticker_list=['AAPL','MSFT','AMZN','TSLA','GOOG','META','TSM','NVDA','NFLX','AMD']
ticker=st.sidebar.selectbox('Select Company',ticker_list)

data= yf.download(ticker,start=sdate,end=edate)
data.insert(0,'Date',data.index,True)
data.reset_index(drop=True,inplace=True)
st.write('Data from',sdate,'to',edate)
st.write(data)

st.header('Data Visualization')
st.subheader('Plot of the Data')
fig=px.line(data,x='Date',y=data.columns,title='Stock Price of Selected Company',template='plotly_dark',width=950,height=600)
st.plotly_chart(fig)

column=st.selectbox('Select the Column to Forecast',data.columns[1:])

data=data[['Date',column]]
st.write('Selected Column')
st.write(data)

st.header('Is Data Stationary?')
st.write(adfuller(data[column])[1]<0.05)


