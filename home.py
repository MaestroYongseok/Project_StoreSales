# -*- coding:utf-8 -*-

import streamlit as st
from PIL import Image
import pandas as pd

def run_home():
    img = Image.open("image/mak.png")
    st.image(img)

    st.markdown("<h1 style='color: red;'>Store Sales - Time Series Forecasting</h1>", unsafe_allow_html=True)
    st.markdown("Member|Skills|GitHub & Blog \n |:--:|:--:|:--:| \n |Jae-Myoung Choi|Analysis & Preprocessing | ![git](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FJVyru%2FbtseBexHPj3%2FVcOibIVPqoCkgLTA8mpP61%2Fimg.png) : https://github.com/ChoiJMS2m  ![blog](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Ftc61f%2FbtseGWwjM0C%2FD4Su9TE8awMKMdyhstKAO0%2Fimg.jpg) : https://james-choi88.tistory.com| \n |Yong-Seok Kwon|Analysis & Dashboard|![git](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FJVyru%2FbtseBexHPj3%2FVcOibIVPqoCkgLTA8mpP61%2Fimg.png) : https://github.com/MaestroYongseok  ![Blog](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FyQsGh%2FbtseAUfuBOB%2FCt0quqF2Ue8OMoqIOUhTvk%2Fimg.png) : https://blog.naver.com/maestrokwon78| \n |Yong-Jun Yoon|Dashboard & PPT|![git](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FJVyru%2FbtseBexHPj3%2FVcOibIVPqoCkgLTA8mpP61%2Fimg.png) : https://github.com/yunyj1998  ![Blog](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FyQsGh%2FbtseAUfuBOB%2FCt0quqF2Ue8OMoqIOUhTvk%2Fimg.png) : https://blog.naver.com/yunyj1998|")

    st.markdown("### Analytic Language Tools")
    col1, col2, col3 = st.columns(3)
    with col1:
        img1 = Image.open("image/st1.png")
        st.image(img1, width=300)

    with col2:
        img2 = Image.open("image/py1.png")
        st.image(img2, width=300)

    with col3:
        img3 = Image.open("image/py2.png")
        st.image(img3, width=300)

    st.markdown("### Project Overview\n\n"
                "- Use time series forecasting to predict store sales based on data from Corporación Favorita, a large grocery retailer based in Ecuador \n\n"
                "- Built a model to more accurately predict selling unit costs for thousands of items sold in multiple Favorita stores \n\n"
                "- To practice machine learning skills with an easy-to-access training dataset consisting of dates, store and item information, promotions, and sales unit prices.")

    st.markdown("### Duration \n"
                "- 2023.04.20~05.17")

    st.markdown("### Purpose of analysis")
    st.markdown(
        " - To gain insight into the store's historical sales data and use that information to predict future sales trends\n\n"
        " - Stores make more informed decisions about inventory management, marketing strategies, and overall business planning\n\n"
        " - By analyzing patterns and trends in sales data, project teams identify factors that may be contributing to changes in sales, such as seasonality or external events\n\n"
        " - Ultimately providing actionable insights to help stores optimize operations and improve financial performance.")

    st.markdown("### Identify and forecast sales changes")
    st.markdown(" - We chose :blue[earthquakes, regions, times, holidays, events, and oil prices] as variables for changes in sales. \n\n"
                " - We will look at how sales change for each variable.")

    st.markdown("### Model the final forecast")
    st.markdown(" - Regression analysis and time series analysis were used to determine the prediction model.")
    col1, col2 = st.columns(2)
    pass


