# -*- coding:utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils import load_data

import streamlit as st

def total_Sales(train, stores):
    st.title("Total Sales")
    st.markdown("""
         <div style="color:blue;display:fill;border-radius:8px;
             background-color:#323232;font-size:150%;
             font-family:Nexa;letter-spacing:0.5px">
         <p style="padding: 8px;color:white;"><b>Viz 1.There is a graph that shows us daily Total Sales Across All Stores.  </b></p>
     </div>
    """, unsafe_allow_html=True)
    st.markdown("""
       - Total sales from 2015 to 2016 are showing a slight increase with a certain pattern of repetition.
     - Also, there is a significant decrease in sales at the end of the year or the beginning of the year.
     - It is estimated that all stores are closed on New Year's holidays or New Year's Eve, so there are no sales.
     - Noted that the earthquake marked with the red dotted band seems to boost the sales few days after.""")
    st.markdown("\n")
    # 데이터 타입 변환
    train.onpromotion = train.onpromotion.astype("float16")
    train.sales = train.sales.astype("float32")
    stores.cluster = stores.cluster.astype("int8")
    train['date'] = pd.to_datetime(train['date'])
    train = train.set_index('date')
    train = train.drop('id', axis=1)
    train[['store_nbr', 'family']] = train[['store_nbr', 'family']].astype('category')
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.set(title="'Total Sales Across All Stores")
    total_sales = train.sales.groupby("date").sum()
    ax.plot(total_sales)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.axvline(x=pd.Timestamp('2016-04-16'), color='r', linestyle='--', linewidth=4, alpha=0.3)
    ax.text(pd.Timestamp('2016-04-20'), 1400000, 'The Earthquake', rotation=360, c='r')
    plt.xticks(rotation=70)
    st.pyplot(fig)
    st.markdown("📌 **Interpret:**\n"
      "- The earthquake occurred on April 16, 2016, and several aftershocks occurred. \n"
       "- You can see a sharp increase in sales in stores in areas affected by the earthquake or in areas where you felt a great deal of shaking."
      )
    st.markdown("\n")
    # 각 스토어의 일일 총 판매량 딕셔너리로 만들기
    daily_sale_dict = {}
    for i in train.store_nbr.unique():
        daily_sale = train[train['store_nbr'] == i]
        daily_sale_dict[i] = daily_sale
    # 위의 결과를 시각화하는데 지진 표시되게 함
    fig = plt.figure(figsize=(30, 30))
    for i in daily_sale_dict.keys():
        plt.subplot(8, 7, i)
        plt.title('Store {} sale'.format(i))
        plt.tight_layout(pad=5)
        sale = daily_sale_dict[i].sales
        sale.plot()
        plt.axvline(x=pd.Timestamp('2016-04-16'), color='r', linestyle='--', linewidth=2, alpha=0.3)  # mark the earthquake
    st.pyplot(fig)
    st.markdown(" 📌 **Interpret:**\n" 
                 "- There are so many gaps or white space in some stores sales record. I guess they are from either temporary store close or the store not yet operate.\n"
                 "- There will be a sharp spike in sales (ex. store # 35 which had 3 spikes around the year's end of 2014, 2015, 2016)\n"
                 "- Considering the earthquake, it affected the sales variedly from store to store. Some store such as 18,20,21,etc.. saw a great one time spike.\n"
                 "- Some store such as store 5,26,35,etc.. didn't have any huge sales changes. However, for store 53, it's the gamechanger. Store 53 had gone through a long duration of sale increase for years.\n"
                 "- The main theme here is that, on average, store sales are in uptrend since 2013."
                )
    st.markdown("\n")

    # 제품군별 매출 딕셔너리로 만들기
    by_fam_dic = {}
    fam_list = train.family.unique()

    for fam in fam_list:
        by_fam_dic[fam] = train[train['family'] == fam].sales

    # 위의 결과를 시각화하는데 지진 표시되게 함
    fig = plt.figure(figsize=(30, 50))

    for i, fam in enumerate(by_fam_dic.keys()):
        plt.subplot(11, 3, i + 1)
        plt.title('{} sale'.format(fam))
        plt.tight_layout(pad=5)
        sale = by_fam_dic[fam]
        sale.plot()
        plt.axvline(x=pd.Timestamp('2016-04-16'), color='r', linestyle='--', linewidth=2, alpha=0.3)  # mark the earthquake

    st.pyplot(fig)
    st.markdown("📌 **Interpret:**\n"
                "- Each family has their own selling paterns.\n"
                "- However, Frozen Food and School and Office Supplies shown highly seasonal cycle.\n"
                "- Frozen Food : Sell more on New Year\n"
                "- School and Office Supplies: Sale more around AUG")