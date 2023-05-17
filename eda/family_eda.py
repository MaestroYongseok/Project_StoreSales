# -*- coding:utf-8 -*-

import pandas as pd
import streamlit as st
from utils import load_data, date_select

from statsmodels.tsa.stattools import adfuller
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
from pingouin import ttest

def familyShow(train, transactions):
    df_data = train.merge(transactions, how='left', on=['date', 'store_nbr'])

    df_data.date = pd.to_datetime(df_data.date)
    df_data['year'] = df_data['date'].dt.year
    df_data['month'] = df_data['date'].dt.month
    df_data['week'] = df_data['date'].dt.isocalendar().week
    df_data['quarter'] = df_data['date'].dt.quarter
    df_data['day_of_week'] = df_data['date'].dt.day_name()

    st.markdown('### Show about family')

    color1 = st.color_picker('Pick A Color', '#00f900')
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['year', 'month', 'week', 'quarter', 'store_nbr'])
    with tab1:
        year = st.selectbox('year:', df_data['year'].unique())
        family_df1 = np.round(
            df_data.loc[df_data['year'] == year, :].groupby('family')['sales'].agg('mean').sort_values(ascending=False),
            1)

        fig1 = go.Figure(go.Bar(
            x=family_df1.values,
            y=family_df1.index,
            orientation='h',
            marker=dict(
                color=color1
            )
        ))

        # Customize the layout
        fig1.update_layout(
            title=f'Sales for {year} year',
            xaxis_title='Mean of Sales',
            yaxis_title='Family',
            plot_bgcolor='white',
            width=800,
            height=600,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig1)
    with tab2:
        month = st.selectbox('Month:', df_data['month'].unique())
        family_df2 = np.round(
            df_data.loc[df_data['month'] == month, :].groupby('family')['sales'].agg('mean').sort_values(
                ascending=False),
            1)


        fig2 = go.Figure(go.Bar(
            x=family_df2.values,
            y=family_df2.index,
            orientation='h',
            marker=dict(
                color=color1
            )
        ))

        # Customize the layout
        fig2.update_layout(
            title=f'Sales for {month} month',
            xaxis_title='Mean of Sales',
            yaxis_title='Family',
            plot_bgcolor='white',
            width=800,
            height=600,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig2)

    with tab3:
        week = st.selectbox('week:', df_data['week'].unique())
        family_df3 = np.round(
            df_data.loc[df_data['week'] == week, :].groupby('family')['sales'].agg('mean').sort_values(
                ascending=False),
            1)

        fig3 = go.Figure(go.Bar(
            x=family_df3.values,
            y=family_df3.index,
            orientation='h',
            marker=dict(
                color=color1
            )
        ))

        # Customize the layout
        fig3.update_layout(
            title=f'Sales for {week} week',
            xaxis_title='Mean of Sales',
            yaxis_title='Family',
            plot_bgcolor='white',
            width=800,
            height=600,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig3)
    with tab4:
        quarter = st.selectbox('quarter:', df_data['quarter'].unique())
        family_df4 = np.round(
            df_data.loc[df_data['quarter'] == quarter, :].groupby('family')['sales'].agg('mean').sort_values(
                ascending=False),
            1)

        fig4 = go.Figure(go.Bar(
            x=family_df4.values,
            y=family_df4.index,
            orientation='h',
            marker=dict(
                color=color1
            )
        ))

        # Customize the layout
        fig4.update_layout(
            title=f'Sales for {quarter} quarter',
            xaxis_title='Mean of Sales',
            yaxis_title='Family',
            plot_bgcolor='white',
            width=800,
            height=600,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig4)
    with tab5:
        store_nbr_options = sorted(df_data['store_nbr'].unique())
        store_nbr = st.selectbox('store_nbr:', store_nbr_options, index=0)
        # store_nbr = st.selectbox('store_nbr:', df_data["store_nbr"].unique())
        family_df5 = np.round(
            df_data.loc[df_data['store_nbr'] == store_nbr, :].groupby('family')['sales'].agg('mean').sort_values(
                ascending=False), 1)

        fig5 = go.Figure(go.Bar(
            x=family_df5.values,
            y=family_df5.index,
            orientation='h',
            marker=dict(
                color=color1
            )
        ))

        # Customize the layout
        fig5.update_layout(
            title=f'Sales for Store : No.{store_nbr}',
            xaxis_title='Mean of Sales',
            yaxis_title='Family',
            plot_bgcolor='white',
            width=800,
            height=600,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig5)



def run_Show():
    train, stores, oil, transactions, holidays_events = load_data()

    submenu = st.sidebar.selectbox("Submenu", ['familyShow'])
    if submenu == 'familyShow':
        st.markdown("## To show the Sales about family")
        familyShow(train, transactions)
    else:
        pass