
import streamlit as st
import pickle
import numpy as np
import pandas as pd

from PIL import Image

import streamlit_authenticator as stauth
import yaml






@st.cache()



def get_original_data():
    return pd.read_csv("Capstone Backlog Data.csv")

def get_agg_data():
    return pd.read_csv("data_agg.csv")




def main():
    st.set_page_config(
    page_title="Real Time Supply/Demand Forecasting Model",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.avnet_url_placeholder/help',
        'Report a bug': "https://www.avnet_url_placeholder/bug"
    })
    
    # print(df.head())
    

    st.title("_Real-Time_ :red[ Supply/Demand ]        _Forecasting           Dashboard_")
    
    image = Image.open('avnet.jpg')
    st.image(image,width=800)
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized'])
    name, authentication_status, username = authenticator.login('Login', 'main')
    

    if st.session_state["authentication_status"]:
        authenticator.logout('Logout', 'main')
        st.write(f'Welcome *{st.session_state["name"]}*')
        role = authenticator.credentials["usernames"][username]['role']
        st.session_state['role'] = role
        st.session_state['user_id'] = username
        st.write(f'Your user role is *{st.session_state["role"]}*. Contact admin to change roles.')
        
            
    elif st.session_state["authentication_status"] == False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] == None:
        st.warning('Please enter your username and password')
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)
        

       
if __name__=='__main__':
    main()
