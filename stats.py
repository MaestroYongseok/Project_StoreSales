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
from arima import run_arima

def adfullerTest(oil):
    st.markdown("### Augmented Dickey-Fuller Test(ADF) \n"
                "#### What is ADF? \n"
                "- The Augmented Dickey-Fuller test is a type of statistical test called a unit root test. In probability theory and statistics, a unit root is a feature of some stochastic processes (such as random walks) that can cause problems in statistical inference involving time series models. In simple terms, the unit root is non-stationary but does not always have a trend component. \n"
                "- The unit root test is then carried out under the null hypothesis $\gamma = 0$.  Once a value for the test statistic")

    st.latex(r'''{\displaystyle \mathrm {DF} _{\tau }={\frac {\hat {\gamma }}{\operatorname {SE} ({\hat {\gamma }})}}}''')
    st.markdown(r"- is computed it can be compared to the relevant critical value for the Dickey–Fuller test. As this test is asymmetrical, we are only concerned with negative values of our test statistic ${\displaystyle \mathrm {DF} _{\tau }}$ If the calculated test statistic is less (more negative) than the critical value, then the null hypothesis of $\gamma = 0$ is rejected and no unit root is present.", unsafe_allow_html=True)
    st.markdown("#### Hypothesis \n"
                "- ADF test is conducted with the following assumptions: \n"
                "   + Null Hypothesis ($H_0$): Series is non-stationary, or series has a unit root. \n"
                "   + Alternate Hypothesis($H_A$): Series is stationary, or series has no unit root. \n"
                "- If Test statistic < Critical Value and $p-value < 0.05$\n"
                "   + Then, Reject Null Hypothesis($H_0$), i.e., time series does not have a unit root, meaning it is stationary. It does not have a time-dependent structure."
                )

    # Plot the results
    fig = px.line(oil, x="date", y="dcoilwtico")
    fig.update_layout(
        title="Augmented Dickey-Fuller Test Results for Oil Prices",
        xaxis_title="Date",
        yaxis_title="dcoilwtico",
    )
    st.plotly_chart(fig)

    oil = date_select(oil, col='date').dropna()
    adf_test = adfuller(oil["dcoilwtico"], autolag="AIC")
    dfoutput = pd.Series(adf_test[0:4], index=['Test Statistic', 'p-value', '#lags used', 'number of observations used'])
    for key, value in adf_test[4].items():
        dfoutput['critical value (%s)' % key] = value
    dfoutput = dfoutput.reset_index()
    dfoutput.columns = ['index', 'result']

    if adf_test[1] > 0.05:
        st.markdown(":green[$H_0$] : **Series is non-stationary, or series has a unit root.**")
    else:
        st.markdown(":green[$H_1$] : **Series is stationary, or series has no unit root.**")
    st.dataframe(dfoutput, use_container_width=True)

def twoMeans(train, transactions):
    st.markdown("- **The independent samples $t$-test** comes in two different forms, Student’s and Welch’s. The original Student **$t$-test** – which is the one I’ll describe in this section – is the simpler of the two, but relies on much more restrictive assumptions than the Welch **$t$-test**. Assuming for the moment that you want to run a two-sided test, the goal is to determine whether two “independent samples” of data are drawn from populations with the same mean (the null hypothesis) or different means (the alternative hypothesis). When we say “independent” samples, what we really mean here is that there’s no special relationship between observations in the two samples. This probably doesn’t make a lot of sense right now, but it will be clearer when we come to talk about the paired samples **$t$-test** later on. For now, let’s just point out that if we have an experimental design where participants are randomly allocated to one of two groups, and we want to compare the two groups’ mean performance on some outcome measure, then an independent samples **$t$-test** (rather than a paired samples **$t$-test**) is what we’re after. \n"
                r"- Okay, so let’s let $\mu_1$ denote the true population mean for group 1 , and $\mu_2$ will be the true population mean for group 2, and as usual we’ll let $\bar{X}_1$ and $\bar{X}_2$ denote the observed sample means for both of these groups. Our null hypothesis states that the two population means are identical ($\mu_1 = \mu_2$) and the alternative to this is that they are not ($\mu_1 \neq \mu_2$). Written in Written in mathematical-ese, this is…")
    st.latex(r"""
        \begin{split}
        \begin{array}{ll}
        H_0: & \mu_1 = \mu_2  \\
        H_1: & \mu_1 \neq \mu_2
        \end{array}
        \end{split}
    """)

    mu1 = 0
    sigma = 1
    mu2 = 2

    x1 = np.linspace(mu1 - 4 * sigma, mu1 + 4 * sigma, 100)
    y1 = 100 * stats.norm.pdf(x1, mu1, sigma)
    x2 = np.linspace(mu2 - 4 * sigma, mu2 + 4 * sigma, 100)
    y2 = 100 * stats.norm.pdf(x2, mu2, sigma)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    sns.lineplot(x=x1, y=y1, color='black', ax=ax1)

    sns.lineplot(x=x1, y=y1, color='black', ax=ax2)
    sns.lineplot(x=x2, y=y2, color='black', ax=ax2)

    ax1.text(0, 43, 'null hypothesis', size=20, ha="center")
    ax2.text(0, 43, 'alternative hypothesis', size=20, ha="center")

    ax1.set_frame_on(False)
    ax2.set_frame_on(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax1.axhline(y=0, color='black')
    ax2.axhline(y=0, color='black')

    st.pyplot(fig)

    st.markdown('<hr>', unsafe_allow_html=True)
    df_data = train.merge(transactions, how='left', on=['date','store_nbr'])

    df_data.date = pd.to_datetime(df_data.date)
    df_data['year'] = df_data['date'].dt.year
    df_data['month'] = df_data['date'].dt.month
    df_data['week'] = df_data['date'].dt.isocalendar().week
    df_data['quarter'] = df_data['date'].dt.quarter
    df_data['day_of_week'] = df_data['date'].dt.day_name()

    st.markdown('### Data Visualization')
    month = st.sidebar.selectbox('Month:', df_data['month'].unique())
    family_df = np.round(df_data.loc[df_data['month']==month, :].groupby('family')['sales'].agg('mean').sort_values(ascending=False), 1)

    color1 = st.color_picker('Pick A Color', '#00f900')
    fig = go.Figure(go.Bar(
        x = family_df.values,
        y = family_df.index,
        orientation='h',
        marker=dict(
            color=color1
        )
    ))

    # Customize the layout
    fig.update_layout(
        title=f'Sales for {month} month',
        xaxis_title='Mean of Sales',
        yaxis_title='Family',
        plot_bgcolor='white',
        width=800,
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig)

    st.markdown("### Data Cleaning and Comparison")
    st.markdown("![](https://www.investopedia.com/thmb/R4twtcFq0xh1C2YLF-_QfeH30Is=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/ttest2-147f89de0b384314812570db74f16b17.png)")
    col1 = st.selectbox('Select Column 1', df_data['family'].unique(), key="family_col1")
    col2 = st.selectbox('Select Column 2', df_data['family'].unique(), key="family_col2")
    if col1 != col2:
        st.markdown("Sample Size is Equal?")
        new_df = df_data.loc[(df_data['month']==month) & (df_data['family'].isin([col1, col2])), :].reset_index(drop=True)
        summary_df = np.round(new_df.groupby('family').agg({'family':'size', 'sales':'mean'}), 1)
        st.dataframe(summary_df)
        st.markdown("Independent Test")

        wide_df = pd.DataFrame({
            col1 : new_df.loc[new_df['family'] == col1]['sales'],
            col2 : new_df.loc[new_df['family'] == col2]['sales']
        })

        result = ttest(wide_df[col1], wide_df[col2], correction=False)
        st.dataframe(result)
        if result['p-val'].values > 0.05:
            st.markdown(":green[$H_0$] : **The means for the two sales are equal.**")
            st.markdown("<hr>", unsafe_allow_html=True)
            grouped = new_df.groupby('family')
            means = grouped['sales'].agg('mean')

            # Define the data for the plot
            data = pd.DataFrame({'family': means.index, 'sales': means.values})
            # Create the Plotly bar plot
            fig = px.bar(data, x='sales', y='family', orientation='h', title='Average Sales by Family',
                         labels={'sales': 'Sales($)', 'species': 'Family'})

            # Customize the layout
            fig.update_layout(
                plot_bgcolor='white',
                width=800,
                height=400
            )

            st.plotly_chart(fig)
        else:
            st.markdown(":green[$H_1$] : **The means for the two populations are not equal.**")
            st.markdown("<hr>", unsafe_allow_html=True)
            grouped = new_df.groupby('family')
            means = grouped['sales'].agg('mean')

            # Define the data for the plot
            data = pd.DataFrame({'family': means.index, 'sales': means.values})
            # Create the Plotly bar plot
            fig = px.bar(data, x='sales', y='family', orientation='h', title='Average Sales by Family',
                         labels={'sales': 'Sales($)', 'species': 'Family'})

            # Customize the layout
            fig.update_layout(
                plot_bgcolor='white',
                width=800,
                height=400
            )

            st.plotly_chart(fig)

    else:
        st.warning("Two Columns are must be different")










def run_stat():
    train, stores, oil, transactions, holidays_events = load_data()

    submenu = st.sidebar.selectbox("Submenu", ['Time Series', 'ARIMA', 'Two Means'])
    if submenu == 'Time Series':
        st.markdown("## Time Series Analysis")
        adfullerTest(oil)
    elif submenu == 'ARIMA':
        run_arima()
    elif submenu == 'Two Means':
        st.markdown("## Two Means")
        twoMeans(train, transactions)