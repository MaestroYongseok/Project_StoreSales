# -*- coding:utf-8 -*-
import streamlit as st
from streamlit_option_menu import option_menu
from description import run_description
from data_intro import run_data
from eda.eda import run_eda
from stats import run_stat
from ml import run_ml
from home import run_home

def main():
    st.set_page_config(layout="wide")
    with st.sidebar:
        selected = option_menu("Main Menu", ['Home', 'Description', 'Data', 'EDA', 'STAT', 'ML'],
                icons=['house', 'card-checklist', 'card-checklist', 'bar-chart', 'clipboard-data', 'clipboard-data'],
                menu_icon="cast", default_index=0, orientation = 'vertical', key='main_option')

    if selected == 'Home':
        run_home()
    elif selected == 'Description':
        run_description()
    elif selected == 'Data':
        run_data()
    elif selected == 'EDA':
        run_eda()
    elif selected == 'STAT':
        run_stat()
    elif selected == 'ML':
        run_ml()
    else:
        print('error..')


if __name__ == "__main__":
    main()