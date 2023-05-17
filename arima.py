# -*- coding:utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import LinearRegression
from utils import load_data

def run_group(transactions, train):
    transactions['date'] = pd.to_datetime(transactions['date'], format="%Y-%m-%d")
    train['date'] = pd.to_datetime(train['date'], format="%Y-%m-%d")

    transactions = transactions.groupby([pd.Grouper(key='date', freq='W')]).agg(mean=('transactions', 'mean'))
    transactions = transactions.reset_index()
    train_m = train.groupby([pd.Grouper(key='date', freq='M')]).agg(mean=('sales', 'mean'))
    train_m = train_m.reset_index()
    train_w = train.groupby([pd.Grouper(key='date', freq='W')]).agg(mean=('sales', 'mean'))
    train_w = train_w.reset_index()

    train_m['time'] = np.arange(len(train_m.index))
    column_time_m = train_m.pop('time')
    train_m.insert(1, 'time', column_time_m)
    train_w['time'] = np.arange(len(train_w.index))
    column_time_w = train_w.pop('time')
    train_w.insert(1, 'time', column_time_w)

    st.markdown("# DataFrame with Grouping")

    with st.expander("See the DataFrame", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("- Train grouped by week")
            st.dataframe(train_w.head())
        with col2:
            st.markdown("- Train grouped by month")
            st.dataframe(train_m.head())
        with col3:
            st.markdown("- Transactions grouped by week")
            st.dataframe(transactions.head())

    st.markdown("# Grouped Line Graph")

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 20))
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.25)

    # TRANSACTIONS (WEEKLY)
    axes[0].plot('date', 'mean', data=transactions, color='grey', marker='o')
    axes[0].set_title("Transactions (grouped by week)", fontsize=20)

    # SALES (WEEKLY)
    axes[1].plot('time', 'mean', data=train_w, color='0.75')
    axes[1].set_title("Sales (grouped by week)", fontsize=20)
    # linear regression
    axes[1] = sns.regplot(x='time',
                          y='mean',
                          data=train_w,
                          scatter_kws=dict(color='0.75'),
                          ax=axes[1])

    # SALES (MONTHLY)
    axes[2].plot('time', 'mean', data=train_m, color='0.75')
    axes[2].set_title("Sales (grouped by month)", fontsize=20)
    # linear regression
    axes[2] = sns.regplot(x='time',
                          y='mean',
                          data=train_m,
                          scatter_kws=dict(color='0.75'),
                          line_kws={"color": "red"},
                          ax=axes[2])

    st.pyplot(fig)

def run_lag(transactions, train):

    transactions['date'] = pd.to_datetime(transactions['date'], format="%Y-%m-%d")
    train['date'] = pd.to_datetime(train['date'], format="%Y-%m-%d")

    transactions = transactions.groupby([pd.Grouper(key='date', freq='W')]).agg(mean=('transactions', 'mean'))
    transactions = transactions.reset_index()
    train_m = train.groupby([pd.Grouper(key='date', freq='M')]).agg(mean=('sales', 'mean'))
    train_m = train_m.reset_index()
    train_w = train.groupby([pd.Grouper(key='date', freq='W')]).agg(mean=('sales', 'mean'))
    train_w = train_w.reset_index()

    train_m['time'] = np.arange(len(train_m.index))
    column_time_m = train_m.pop('time')
    train_m.insert(1, 'time', column_time_m)
    train_w['time'] = np.arange(len(train_w.index))
    column_time_w = train_w.pop('time')
    train_w.insert(1, 'time', column_time_w)

    grouped_train_m_lag1 = train.groupby([pd.Grouper(key='date', freq='M')]).agg(mean=('sales', 'mean'))
    grouped_train_m_lag1 = grouped_train_m_lag1.reset_index()
    name = 'Lag_' + str(1)
    grouped_train_m_lag1[name] = grouped_train_m_lag1['mean'].shift(1)

    grouped_train_w_lag1 = train.groupby([pd.Grouper(key='date', freq='W')]).agg(mean=('sales', 'mean'))
    grouped_train_w_lag1 = grouped_train_w_lag1.reset_index()
    name = 'Lag_' + str(1)
    grouped_train_w_lag1[name] = grouped_train_w_lag1['mean'].shift(1)

    st.markdown("## DataFrame to Lag")

    with st.expander("See the DataFrame", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("- Train Lag1 grouped by week")
            st.dataframe(grouped_train_m_lag1)
        with col2:
            st.markdown("- Train Lag1 grouped by month")
            st.dataframe(grouped_train_w_lag1)

    st.markdown("# Lag_1 Graph")

    fig1, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.35)
    axes[0].plot('Lag_1', 'mean', data=grouped_train_w_lag1, color='0.75', linestyle=(0, (1, 10)))
    axes[0].set_title("Sales (grouped by week)", fontsize=20)
    axes[0] = sns.regplot(x='Lag_1',
                          y='mean',
                          data=grouped_train_w_lag1,
                          scatter_kws=dict(color='0.75'),
                          ax=axes[0])

    axes[1].plot('Lag_1', 'mean', data=grouped_train_m_lag1, color='0.75', linestyle=(0, (1, 10)))
    axes[1].set_title("Sales (grouped by month)", fontsize=20)
    axes[1] = sns.regplot(x='Lag_1',
                          y='mean',
                          data=grouped_train_m_lag1,
                          scatter_kws=dict(color='0.75'),
                          line_kws={"color": "red"},
                          ax=axes[1])

    st.pyplot(fig1)

def run_moving_average(transactions, train):
    st.markdown("# Moving Average")
    transactions['date'] = pd.to_datetime(transactions['date'], format="%Y-%m-%d")
    train['date'] = pd.to_datetime(train['date'], format="%Y-%m-%d")

    transactions = transactions.groupby([pd.Grouper(key='date', freq='W')]).agg(mean=('transactions', 'mean'))
    transactions = transactions.reset_index()
    train_m = train.groupby([pd.Grouper(key='date', freq='M')]).agg(mean=('sales', 'mean'))
    train_m = train_m.reset_index()
    train_w = train.groupby([pd.Grouper(key='date', freq='W')]).agg(mean=('sales', 'mean'))
    train_w = train_w.reset_index()

    train_m['time'] = np.arange(len(train_m.index))
    column_time_m = train_m.pop('time')
    train_m.insert(1, 'time', column_time_m)
    train_w['time'] = np.arange(len(train_w.index))
    column_time_w = train_w.pop('time')
    train_w.insert(1, 'time', column_time_w)

    moving_average_tran = transactions['mean'].rolling(window=7, center=True, min_periods=4).mean()
    moving_average_train_w = train_w['mean'].rolling(window=7, center=True, min_periods=4).mean()
    moving_average_train_m = train_m['mean'].rolling(window=7, center=True, min_periods=4).mean()

    fig2, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 20))
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.15)

    axes[0] = transactions['mean'].plot(color='0.75', linestyle='dashdot', ax=axes[0])
    axes[0] = moving_average_tran.plot(linewidth=3, color='g', ax=axes[0])
    axes[0].set_title('Transactions Moving Average', fontsize=18)

    axes[1] = train_w['mean'].plot(color='0.75', linestyle='dashdot', ax=axes[1])
    axes[1] = moving_average_train_w.plot(linewidth=3, color='g', ax=axes[1])
    axes[1].set_title('Sales Moving Average (Week)', fontsize=18)

    axes[2] = train_m['mean'].plot(color='0.75', linestyle='dashdot', ax=axes[2])
    axes[2] = moving_average_train_m.plot(linewidth=3, color='g', ax=axes[2])
    axes[2].set_title('Sales Moving Average (Month)', fontsize=18)
    st.pyplot(fig2)

def run_trend(transactions, train):
    st.markdown("# Linear Trend")
    transactions['date'] = pd.to_datetime(transactions['date'], format="%Y-%m-%d")
    train['date'] = pd.to_datetime(train['date'], format="%Y-%m-%d")

    transactions = transactions.groupby([pd.Grouper(key='date', freq='W')]).agg(mean=('transactions', 'mean'))
    transactions = transactions.reset_index()
    train_m = train.groupby([pd.Grouper(key='date', freq='M')]).agg(mean=('sales', 'mean'))
    train_m = train_m.reset_index()
    train_w = train.groupby([pd.Grouper(key='date', freq='W')]).agg(mean=('sales', 'mean'))
    train_w = train_w.reset_index()

    train_m['time'] = np.arange(len(train_m.index))
    column_time_m = train_m.pop('time')
    train_m.insert(1, 'time', column_time_m)
    train_w['time'] = np.arange(len(train_w.index))
    column_time_w = train_w.pop('time')
    train_w.insert(1, 'time', column_time_w)

    fig3, axes = plt.subplots(nrows=4, ncols=1, figsize=(20, 15))
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.45)

    transactions['date'] = pd.to_datetime(transactions['date'], format="%Y-%m-%d")
    dp_tran = DeterministicProcess(index=transactions['date'], constant=True, order=1, drop=True)
    dp_tran.index.freq = 'W'

    train_w['date'] = pd.to_datetime(train_w['date'], format="%Y-%m-%d")
    dp_train_w = DeterministicProcess(index=train_w['date'], constant=True, order=1, drop=True)
    dp_train_w.index.freq = 'W'  # manually set the frequency of the index
    # 'in_sample' creates features for the dates given in the `index` argument
    X1 = dp_tran.in_sample()
    y1 = transactions["mean"]  # the target
    y1.index = X1.index

    X3 = dp_train_w.in_sample()
    y3 = train_w["mean"]  # the target
    y3.index = X3.index
    # The intercept is the same as the `const` feature from
    # DeterministicProcess. LinearRegression behaves badly with duplicated
    # features, so we need to be sure to exclude it here.
    model = LinearRegression(fit_intercept=False)
    model.fit(X1, y1)
    y1_pred = pd.Series(model.predict(X1), index=X1.index)
    axes[0] = y1.plot(linestyle='dashed', label="mean", color="0.75", ax=axes[0], use_index=True)
    axes[0] = y1_pred.plot(linewidth=3, label="Trend", color='b', ax=axes[0], use_index=True)
    axes[0].set_title("Transactions Linear Trend", fontsize=18)
    _ = axes[0].legend()

    model1 = LinearRegression(fit_intercept=False)
    model1.fit(X3, y3)
    y3_pred = pd.Series(model1.predict(X3), index=X3.index)
    axes[2] = y3.plot(linestyle='dashed', label="mean", color="0.75", ax=axes[2], use_index=True)
    axes[2] = y3_pred.plot(linewidth=3, label="Trend", color='b', ax=axes[2], use_index=True)
    axes[2].set_title("Sales Linear Trend", fontsize=18)
    _ = axes[2].legend()

    # forecast Trend for future 30 steps
    steps = 30
    X2 = dp_tran.out_of_sample(steps=steps)
    y2_fore = pd.Series(model.predict(X2), index=X2.index)
    y2_fore.head()
    axes[1] = y1.plot(linestyle='dashed', label="mean", color="0.75", ax=axes[1], use_index=True)
    axes[1] = y1_pred.plot(linewidth=3, label="Trend", color='b', ax=axes[1], use_index=True)
    axes[1] = y2_fore.plot(linewidth=3, label="Predicted Trend", color='r', ax=axes[1], use_index=True)
    axes[1].set_title("Transactions Linear Trend Forecast", fontsize=18)
    _ = axes[1].legend()

    X4 = dp_train_w.out_of_sample(steps=steps)
    y4_fore = pd.Series(model1.predict(X4), index=X4.index)
    y4_fore.head()
    axes[3] = y3.plot(linestyle='dashed', label="mean", color="0.75", ax=axes[3], use_index=True)
    axes[3] = y3_pred.plot(linewidth=3, label="Trend", color='b', ax=axes[3], use_index=True)
    axes[3] = y4_fore.plot(linewidth=3, label="Predicted Trend", color='r', ax=axes[3], use_index=True)
    axes[3].set_title("Sales Linear Trend Forecast", fontsize=18)
    _ = axes[3].legend()

    st.pyplot(fig3)

def run_forecast(transactions, train):
    st.markdown("# Seasonal Forecast")
    transactions['date'] = pd.to_datetime(transactions['date'], format="%Y-%m-%d")
    train['date'] = pd.to_datetime(train['date'], format="%Y-%m-%d")

    transactions = transactions.groupby([pd.Grouper(key='date', freq='W')]).agg(mean=('transactions', 'mean'))
    transactions = transactions.reset_index()
    train_m = train.groupby([pd.Grouper(key='date', freq='M')]).agg(mean=('sales', 'mean'))
    train_m = train_m.reset_index()
    train_w = train.groupby([pd.Grouper(key='date', freq='W')]).agg(mean=('sales', 'mean'))
    train_w = train_w.reset_index()

    train_m['time'] = np.arange(len(train_m.index))
    column_time_m = train_m.pop('time')
    train_m.insert(1, 'time', column_time_m)
    train_w['time'] = np.arange(len(train_w.index))
    column_time_w = train_w.pop('time')
    train_w.insert(1, 'time', column_time_w)

    fig4, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 15))
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.20)

    fourier = CalendarFourier(freq="A", order=10)  # 10 sin/cos pairs for "A"nnual seasonality

    transactions['date'] = pd.to_datetime(transactions['date'], format="%Y-%m-%d")
    dp_tran = DeterministicProcess(index=transactions['date'], constant=True, order=1,
                                   period=None,
                                   seasonal=True,
                                   additional_terms=[fourier], drop=True)
    dp_tran.index.freq = 'W'

    train_w['date'] = pd.to_datetime(train_w['date'], format="%Y-%m-%d")
    dp_train_w = DeterministicProcess(index=train_w['date'], constant=True, order=1,
                                      period=None,
                                      seasonal=True,
                                      additional_terms=[fourier], drop=True)
    dp_train_w.index.freq = 'W'  # manually set the frequency of the index

    # 'in_sample' creates features for the dates given in the `index` argument
    X5 = dp_tran.in_sample()
    y5 = transactions["mean"]  # the target
    y5.index = X5.index

    X6 = dp_train_w.in_sample()
    y6 = train_w["mean"]  # the target
    y6.index = X6.index

    # The intercept is the same as the `const` feature from
    # DeterministicProcess. LinearRegression behaves badly with duplicated
    # features, so we need to be sure to exclude it here.
    model2 = LinearRegression(fit_intercept=False)
    model2.fit(X5, y5)
    y5_pred = pd.Series(model2.predict(X5), index=X5.index)
    X5_fore = dp_tran.out_of_sample(steps=90)
    y5_fore = pd.Series(model2.predict(X5_fore), index=X5_fore.index)

    model3 = LinearRegression(fit_intercept=False)
    model3.fit(X6, y6)
    y6_pred = pd.Series(model3.predict(X6), index=X6.index)
    X6_fore = dp_train_w.out_of_sample(steps=90)
    y6_fore = pd.Series(model3.predict(X6_fore), index=X6_fore.index)

    axes[0] = y5.plot(linestyle='dashed', style='.', label="init mean values", color="0.4", ax=axes[0], use_index=True)
    axes[0] = y5_pred.plot(linewidth=3, label="Seasonal", color='b', ax=axes[0], use_index=True)
    axes[0] = y5_fore.plot(linewidth=3, label="Seasonal Forecast", color='r', ax=axes[0], use_index=True)
    axes[0].set_title("Transactions Seasonal Forecast", fontsize=18)
    _ = axes[0].legend()

    axes[1] = y6.plot(linestyle='dashed', style='.', label="init mean values", color="0.4", ax=axes[1], use_index=True)
    axes[1] = y6_pred.plot(linewidth=3, label="Seasonal", color='b', ax=axes[1], use_index=True)
    axes[1] = y6_fore.plot(linewidth=3, label="Seasonal Forecast", color='r', ax=axes[1], use_index=True)
    axes[1].set_title("Sales Seasonal Forecast", fontsize=18)
    _ = axes[1].legend()

    st.pyplot(fig4)

def run_arima():
    train, stores, oil, transactions, holidays_events = load_data()

    submenu = st.sidebar.selectbox("Submenu2", [ 'Grouping', 'Lag1', 'Moving Average', 'linear Trend', 'Seasonal Forcast'])
    if submenu == 'Grouping':
        run_group(transactions, train)
    elif submenu == 'Lag1':
        run_lag(transactions, train)
    elif submenu == 'Moving Average':
        run_moving_average(transactions, train)
    elif submenu == 'linear Trend':
        run_trend(transactions, train)
    elif submenu == 'Seasonal Forcast':
        run_forecast(transactions, train)
    else:
        pass