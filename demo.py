import streamlit as st

import pandas as pd
import numpy as np

import seaborn as sns
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

app='stock market forecast'
st.write(app)
print("heello")


