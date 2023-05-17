# -*- coding:utf-8 -*-
import pandas as pd
import calendar

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st
from pathlib import Path

from utils import date_select, load_data

import pandas_profiling
import streamlit as st

from eda.family_eda import run_Show
from eda.earthquake_sales import total_Sales
from eda.oil import run_oil_sales
from utils import load_data


@st.cache_resource(experimental_allow_widgets=True)



def show_chart(train, stores, oil, transactions, holidays_events):

    st.markdown("""
    <div style="color:white;display:fill;border-radius:8px;
            background-color:#323232;font-size:150%;
            font-family:Nexa;letter-spacing:0.5px">
        <p style="padding: 8px;color:white;"><b>Viz 1. Oil Price</b></p>
    </div>
    """, unsafe_allow_html=True)
    oil = date_select(oil, col='date')
    oil = oil.set_index(['date'])
    moving_average_oil = oil.rolling(
        window=365,  # 365-day window
        center=True,  # puts the average at the center of the window
        min_periods=183,  # choose about half the window size
    ).median()  # compute the mean (could also do median, std, min, max, ...)

    oil = oil.reset_index()
    moving_average_oil = moving_average_oil.reset_index()

    moving_average_oil.loc[[0, 1], 'dcoilwtico'] = moving_average_oil.loc[2, 'dcoilwtico']
    moving_average_oil.date = pd.to_datetime(moving_average_oil.date)

    df_yr_oil = oil[['date', 'dcoilwtico']]
    fig = make_subplots(rows=1, cols=1, vertical_spacing=0.08,
                        subplot_titles=("Oil price during time"))
    fig.add_trace(
        go.Scatter(x=df_yr_oil['date'], y=df_yr_oil['dcoilwtico'], mode='lines', fill='tozeroy', fillcolor='#c6ccd8',
                   marker=dict(color='#496595'), name='Oil price'),
        row=1, col=1)
    fig.add_trace(go.Scatter(x=moving_average_oil.date, y=moving_average_oil.dcoilwtico, mode='lines', name='Trend'))
    fig.update_layout(height=350, bargap=0.15,
                      margin=dict(b=0, r=20, l=20),
                      title_text="Oil price trend during time",
                      template="plotly_white",
                      title_font=dict(size=25, color='#8a8d93', family="Lato, sans-serif"),
                      font=dict(color='#8a8d93'),
                      hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                      showlegend=False)
    st.plotly_chart(fig)
    st.markdown(":pencil: **Interpret:**\n" 
    "- As can be seen in the graph above, we can divide the oil price trend into **<span style='color:#F1C40F'>three phases</span>**. The first and last of these, Jan2013-Jul2014 and Jan2015-Jul2107 respectively, show stabilised trends with ups and downs. However, in the second phase, Jul2014-Jan2015, oil prices decrease considerably. \n"
                "- Now, taking into account the issue of missing values for oil price, we are going to fill them by **<span style='color:#F1C40F'>backward fill technique</span>**. That means filling missing values with next data point (Forward filling means fill missing values with previous data", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style="color:white;display:fill;border-radius:8px;
            background-color:#323232;font-size:150%;
            font-family:Nexa;letter-spacing:0.5px">
        <p style="padding: 8px;color:white;"><b>Viz 2. Sales Analysis </b></p>
    </div>
    """, unsafe_allow_html=True)
    train['date'] = pd.to_datetime(train['date'])
    train['week'] = train['date'].dt.isocalendar().week
    train['quarter'] = train['date'].dt.quarter
    train['day_of_week'] = train['date'].dt.day_name()
    # data
    df_m_sa = train.groupby('month').agg({"sales": "mean"}).reset_index()
    df_m_sa['sales'] = round(df_m_sa['sales'], 2)
    df_m_sa['month_text'] = df_m_sa['month'].apply(lambda x: calendar.month_abbr[x])
    df_m_sa['text'] = df_m_sa['month_text'] + ' - ' + df_m_sa['sales'].astype(str)

    df_w_sa = train.groupby('week').agg({"sales": "mean"}).reset_index()
    df_q_sa = train.groupby('quarter').agg({"sales": "mean"}).reset_index()

    # chart color
    df_m_sa['color'] = '#496595'
    df_m_sa['color'][:-1] = '#c6ccd8'
    df_w_sa['color'] = '#c6ccd8'

    # chart
    fig1 = make_subplots(rows=2, cols=2, vertical_spacing=0.08,
                        row_heights=[0.7, 0.3],
                        specs=[[{"type": "bar"}, {"type": "pie"}],
                               [{"colspan": 2}, None]],
                        column_widths=[0.7, 0.3],
                        subplot_titles=("Month wise Avg Sales Analysis", "Quarter wise Avg Sales Analysis",
                                        "Week wise Avg Sales Analysis"))

    fig1.add_trace(go.Bar(x=df_m_sa['sales'], y=df_m_sa['month'], marker=dict(color=df_m_sa['color']),
                         text=df_m_sa['text'], textposition='auto',
                         name='Month', orientation='h'),
                  row=1, col=1)
    fig1.add_trace(go.Pie(values=df_q_sa['sales'], labels=df_q_sa['quarter'], name='Quarter',
                         marker=dict(colors=['#334668', '#496595', '#6D83AA', '#91A2BF', '#C8D0DF']), hole=0.7,
                         hoverinfo='label+percent+value', textinfo='label+percent'),
                  row=1, col=2)
    fig1.add_trace(
        go.Scatter(x=df_w_sa['week'], y=df_w_sa['sales'], mode='lines+markers', fill='tozeroy', fillcolor='#c6ccd8',
                   marker=dict(color='#496595'), name='Week'),
        row=2, col=1)

    # styling
    fig1.update_yaxes(visible=False, row=1, col=1)
    fig1.update_xaxes(visible=False, row=1, col=1)
    fig1.update_xaxes(tickmode='array', tickvals=df_w_sa.week, ticktext=[i for i in range(1, 53)],
                     row=2, col=1)
    fig1.update_yaxes(visible=False, row=2, col=1)
    fig1.update_layout(height=750, bargap=0.15,
                      margin=dict(b=0, r=20, l=20),
                      title_text="Average Sales Analysis",
                      template="plotly_white",
                      title_font=dict(size=25, color='#8a8d93', family="Lato, sans-serif"),
                      font=dict(color='#8a8d93'),
                      hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                      showlegend=False)
    st.plotly_chart(fig1)
    st.markdown("📌 **Interpret:** Highest sales are made in the **<span style='color:#F1C40F'>last quarter</span>** of the year, followed by the third. The one with less saling is the first one.", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div style="color:white;display:fill;border-radius:8px;
            background-color:#323232;font-size:150%;
            font-family:Nexa;letter-spacing:0.5px">
        <p style="padding: 8px;color:white;"><b>Viz 3. Time Series Analysis </b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### **What is Trend ?** \n"
                "The trend component of a time series represents a **<span style='color:#F1C40F'>persistent, long-term change in the mean of the series</span>**. The trend is the slowest-moving part of a series, the part representing the largest time scale of importance. In a time series of product sales, an increasing trend might be the effect of a market expansion as more people become aware of the product year by year. \n"
                "### **Moving Average Plot** \n"
                "To see what kind of trend a time series might have, we can use a moving average plot. To compute a moving average of a time series, we compute the average of the values within a **<span style='color:#F1C40F'>sliding window</span>** of some defined width. Each point on the graph represents the average of all the values in the series that fall within the window on either side. The idea is to **<span style='color:#F1C40F'>smooth out</span>** any short-term **<span style='color:#F1C40F'>fluctuations</span>** in the series so that only long-term changes remain. \n", unsafe_allow_html=True)
    sales = train.groupby('date').agg({"sales": "mean"}).reset_index()
    sales.set_index('date', inplace=True)
    moving_average = sales.rolling(
        window=365,  # 365-day window
        center=True,  # puts the average at the center of the window
        min_periods=183,  # choose about half the window size
    ).mean()  # compute the mean (could also do median, std, min, max, ...)
    moving_average['date'] = sales.index

    fig2 = make_subplots(rows=1, cols=1, vertical_spacing=0.08,
                        subplot_titles=("Sales 365 - Day Moving Average"))
    fig2.add_trace(go.Scatter(x=sales.index, y=sales['sales'], mode='lines', fill='tozeroy', fillcolor='#c6ccd8',
                             marker=dict(color='#334668'), name='365-Day Moving Average'))
    fig2.add_trace(go.Scatter(x=moving_average.date, y=moving_average.sales, mode='lines', name='Trend'))
    fig2.update_layout(height=350, bargap=0.15,
                      margin=dict(b=0, r=20, l=20),
                      title_text="Sales trend during years",
                      template="plotly_white",
                      title_font=dict(size=25, color='#8a8d93', family="Lato, sans-serif"),
                      font=dict(color='#8a8d93'),
                      hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                      showlegend=False)
    st.plotly_chart(fig2)
    st.markdown("📌 **Interpret:** As we can appreeciate, sales has an constantly increasing trend during recorded years.", unsafe_allow_html=True)


def run_eda():
    train, stores, oil, transactions, holidays_events = load_data()

    submenu = st.sidebar.selectbox("Menu", [ 'Charts', 'Family EDA', 'Earthquake and Sales', 'Oil and Sales'])
    if submenu == 'Charts':
        show_chart(train, stores, oil, transactions, holidays_events)
        st.caption("Code From : [Time Series Forecasting Tutorial](https://www.kaggle.com/code/javigallego/time-series-forecasting-tutorial)")
    elif submenu == 'Family EDA':
        run_Show()
    elif submenu == 'Earthquake and Sales':
        total_Sales(train, stores)
    elif submenu == 'Oil and Sales':
        run_oil_sales(train, stores, oil, transactions)
    else:
        pass
