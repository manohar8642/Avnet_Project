import streamlit as st
import pandas as pd
import numpy as np
import plotly as pt
from functionality import get_original_data, mpn_data
import matplotlib.pyplot as plt
import xlrd
import modin.pandas as md
import plotly.graph_objects as go
import openpyxl
import os
import xlsxwriter
from io import BytesIO
import datetime
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Manufacturer Dashboard", page_icon="📈",layout='wide')
st.markdown("# Manufacturer Dashboard")
st.sidebar.header("Manufacturer")

def aggregate_data(df,mpn):
    df = df[df['Mock MPN'].isin(mpn)]
    df['year'] = df['Cust Required date'].dt.strftime('%Y')
    df['month'] = df['Cust Required date'].dt.strftime('%m')
    df_agg = df.groupby(['year','month']).agg({'Order Qty':['sum']}).droplevel(0,axis=1)
    df_agg = df_agg.rename(columns={'sum':'Order Qty'}).reset_index()

    return df_agg
    
def color_fulfilment(val):
    color = 'yellow' if val=='Yes' else 'white'
    return f'background-color: {color}'
def to_excel(df,fig):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    image_data = BytesIO(fig.to_image(format="png"))
    # Write the image to the same sheet as the data.
    worksheet.insert_image(2, 3, 'plotly.png', {'image_data': image_data})
    writer.save()
    processed_data = output.getvalue()
    return processed_data

if (st.session_state["authentication_status"]) and ('authentication_status' in st.session_state):

    if st.session_state['role'] in ['manufacturer','avnet']:
        df =  get_original_data()
        today = datetime.date.today()
        df['Cust Required date'] = df['Cust Required date'].apply(lambda x: xlrd.xldate_as_datetime(x, 0))

        if st.session_state['role'] in ['manufacturer']:

            st.write(f'Welcome *{st.session_state["name"]}*')
            manufacturer = st.session_state['user_id']
        elif st.session_state['role'] in ['avnet']:
            st.write(f'Welcome *{st.session_state["name"]}*. You have admin Privileges')
            manufacturer = st.sidebar.selectbox("Select the Manufacturer  ",df['Mock Manufacturer'].unique())
        
        df = df.where(df['Mock Manufacturer'].str.lower()==manufacturer.lower())

        forecast_button = st.sidebar.radio('Do you wish to forecast demand ?',['No','Yes'])
        st.write()
        if forecast_button=='Yes':
            # Replace the placeholder with nothing
            # image_placeholder.empty()
            mpn_list = df['Mock MPN'].dropna().unique().tolist()
            mpn_list.insert(0,'Select All')
            # [l1.append(i) for i in mpn_list]
            
            mpn = st.sidebar.multiselect("Select the MPN to forecast demand  ",mpn_list)
            if 'Select All' in mpn:
                mpn = df['Mock MPN'].dropna().unique().tolist()
            # df = df
            df_agg = aggregate_data(df,mpn)
            
            model = st.sidebar.selectbox("Select the model ",['Arima','Exponential Smoothing','Prophet'])
            forecast_period = st.sidebar.slider("Period of Foreast ⬡",1, 12, 1, 1)
            if st.sidebar.button("Predict"):
                # Replace the placeholder with nothing
                # image_placeholder.empty()
                try:
                    result, smape,rmse,str1 = mpn_data(df_agg,forecast_period,model)
                    result = result.clip(lower=0)
                    # Create a Plotly figure object
                    fig = go.Figure()


                    # Add the actual time series values as a line chart
                    fig.add_trace(go.Scatter(x=result.index, y=result['Order Qty'], name='Actual', line=dict(color='blue', width=2)))

            

                # Add the forecasted values as a line chart    
                    fig.add_trace(go.Scatter(x=result.index, y=result['Prediction'], name='Forecast', line=dict(color='red', width=2)))


                    # Add the upper bound of the predictions as a line chart
                    fig.add_trace(go.Scatter(x=result.index, y=result['Prediction_U'], name='Forecast_Upper', line=dict(color='rgba(0,128,0,0.2)', dash='dot', width=2)))

                

                    # Add the lower bound of the predictions as a line chart
                    fig.add_trace(go.Scatter(x=result.index, y=result['Prediction_L'], name='Forecast_Lower', line=dict(color='rgba(0,128,0,0.2)', dash='dot', width=2)))

        

                    # Add a shaded region between the upper and lower bounds of the predictions
                    fig.add_trace(go.Scatter(x=result.index.union(result.index[::-1]),
                    y=result['Prediction_U'].append(result['Prediction_L'][::-1]),
                    fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Interval', hoverinfo='skip'))
                # Identify the actual values outside the forecast bounds
                    actual_values_outside_bounds = result[(result['Order Qty'] > result['Prediction_U']) | (result['Order Qty'] < result['Prediction_L'])]

            

                # Add a scatter trace to highlight the actual values outside the bounds
                    fig.add_trace(go.Scatter(
                x=actual_values_outside_bounds.index,
                y=actual_values_outside_bounds['Order Qty'],
                mode='markers',
                marker=dict(
                color='orange',
                size=10,
                line=dict(width=2, color='white')
                ),
                    name='Actuals Outside Confidence Interval',
                    hoverinfo='skip'
                ))
                    # Add hover effects to the chart
                    fig.update_layout(hovermode='x')

        

                    # Add a title and axis labels
                    fig.update_layout(title='Actual vs Forecasted Order Quantity', xaxis_title='Time', yaxis_title='Order Quantity')

                

                    # Customize the appearance of the chart
                    fig.update_layout(
                    plot_bgcolor='white',
                    font=dict(size=8),
                    legend=dict(title='', x=0.01, y=1.0, bgcolor='lightgray', bordercolor='gray'),
                    xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False),
                    hoverlabel=dict(font=dict(size=16), bgcolor='white', bordercolor='gray'),
                    margin=dict(l=90, r=20, t=100, b=80),height=700
                    )

                    # Show the chart
                    
                    forecast_df = result.copy()
                    forecast_df = forecast_df[(~forecast_df['Prediction'].isna()) & (forecast_df['Order Qty'].isna())]
                    col1,col2 =st.columns(2)
                    # col2.metric("SMAPE",round(smape,2))
                    col1.dataframe(forecast_df[['Prediction']])
                    if model =='Arima':
                        col2.write("Auto ARIMA is a statistical algorithm used for time series analysis and forecasting. It automatically selects the optimal parameters for an ARIMA model based on the data, making it a useful tool for non-experts in time series analysis.")
                    elif model =='Prophet':
                        col2.write("Prophet is a time series forecasting model developed by Facebook that is useful for predicting future demand patterns. It uses a decomposable time-series model with trend, seasonality, and holiday components to achieve high accuracy and robustness in handling missing data and outliers.")
                    else:

                        col2.write("Exponential smoothing is a popular time series forecasting method that assigns exponentially decreasing weights over time to give more importance to recent observations. It is useful for short-term forecasting and can handle data with trend and seasonality components")
                    
                    st.write(str1)
                    
                    st.plotly_chart(fig,use_container_width=True)
                    col3, col4 = st.columns(2)
                    result.fillna('-',inplace=True)
                    df_xlsx = to_excel(result.reset_index(),fig)

                    
                    col4.download_button(label='📥 Download Current Result',
                                data=df_xlsx ,
                                file_name= "{}_prediction_report.xlsx".format(today),
                    mime="application/vnd.ms-excel")
                except:
                    st.write('Not Enough Data Points. Select Different MPN') 

                

    
       

    else:
        st.write('You are not authorised to view this page. Contact admin for access')
else:
    st.write('You are not logged in. Log in with your username and password in the home page')