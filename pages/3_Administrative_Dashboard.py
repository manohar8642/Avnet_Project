import streamlit as st
import pandas as pd
import numpy as np
import xlrd
import pydeck as pdk
import plotly.express as px
from geopy.geocoders import Nominatim
import altair as alt
from functionality import revenue
import time
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import plotly.graph_objects as go
import locale as lc
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
from folium import GeoJsonTooltip
from folium import GeoJson
import branca.colormap as cm
from branca.colormap import LinearColormap


geolocator = Nominatim(user_agent="MyApp")

def get_original_data():
    return pd.read_csv("Capstone Backlog Data.csv") 

def display_chart_bar(df,x,y):
    
    c = alt.Chart(df).mark_bar().encode(y=y, x=x, tooltip=[y], color=x,order = alt.Order(
      # Sort the segments of the bars by this field
      y,
      sort='ascending'
    ))
    return c
def plotly_bar_chart(
        df: pd.DataFrame,
        x_axis: str = 'Mock MPN',
        y_axis: str = 'Total_Resale_Value'
    ) -> go.Figure:
    this_chart = go.Figure(
        data=[go.Bar(x=df[x_axis], y=df[y_axis], text = df['lead _time_in months'],textposition = 'outside')])
    this_chart.update_yaxes(title_text=y_axis)
    this_chart.update_xaxes(title_text=x_axis)

    return this_chart

st.set_page_config(page_title="Admin Dashboard", page_icon="📈",layout='wide')
st.markdown("# Admin Dashboard")
st.sidebar.header("Admin")

if st.session_state["authentication_status"] and ('authentication_status' in st.session_state) :

    if st.session_state['role'] =='avnet':
        st.write(f'Welcome *{st.session_state["name"]}*')

        
        df = get_original_data()
        df['Cust Required date'] = df['Cust Required date'].apply(lambda x: xlrd.xldate_as_datetime(x, 0))
        df['year'] = df['Cust Required date'].dt.strftime('%Y')
        df['month'] = df['Cust Required date'].dt.strftime('%m')
        df['revenue'] = df['Unit Price']*df['Order Qty']
        df['revenue'] = df['revenue'].round(0)
        df['lead _time_in months'] =np.round( np.floor(df['Plant Materal - Supplier Lead Time (Key)']/30),0)
        df['lead _time_in months'] =df['lead _time_in months'].fillna(0)
        # col1.button('  MPN Level data  ')
        # col2.button('Manufacture Leaderboard')
        # col3.button('Customer Leaderboard')
        # tab1,tab2,tab3 = st.tabs(['MPN Level data','Manufacture Leaderboard','Customer Leaderboard'])
        # col1,col2,col3 = st.columns(3)
        with st.container():
            col1,col2= st.columns(2)
            choice = col2.radio("Select the factor to view KPI's",[ 'Customer', 'Manufacturer','MPN'],horizontal=True,)
            # placeholder = st.empty()
            # chart = display_chart_bar(df)
            # placeholder.altair_chart(chart, use_container_width=True)
            
            if choice=='MPN':
            
                st.sidebar.header("Select filters for mpn data:")
                mpn_list = df['Mock MPN'].dropna().unique().tolist()
                mpn_list.insert(0,'Select All')
                # [l1.append(i) for i in mpn_list]
                time_frame = st.sidebar.selectbox('Select timeframe to view KPIs',['Yearly','Quarterly','Monthly'],key =1 )

                mpn = st.sidebar.multiselect("Select the MPN to show KPI's ",mpn_list,default='Select All') 
                

                if 'Select All' in mpn:
                    mpn = df['Mock MPN'].dropna().unique().tolist()
                df2 = df.copy()
                df2 = df2[df2['Mock MPN'].isin(mpn)]
                
                # col1 = st.columns(1)
                # placeholder = st.empty()
                try:
                    revenue_val, growth = revenue(df2,mpn,time_frame)
                    col1.metric("Resale Value {}".format(time_frame), "$ {:,} ".format(revenue_val), "{} ".format(growth))
                except:
                    st.write("Select a MPN to view KPI's")
                
                x = st.sidebar.selectbox('Select the location  Feature to view KPI ',['Sold to Country/State','Ship to Country/State','Bill to party Country/State'])

                y_values = st.sidebar.selectbox('Select the aggregated feature to view KPI ',['Order Qty','Remaining qty','Resale value'])
                if y_values=='Resale value':
                    y = 'revenue'
                else:
                    y = y_values
                    st.write("Location wise %s for each MPN"%(y_values))
                st.altair_chart(display_chart_bar(df2,x,y),use_container_width=True)


                ship_to_df = {
                    'Ship to Country/State': ['Arizona', 'Beijing', 'California', 'Chihuahua', 'Florida', 'Hong Kong Island', 'Internationl Country', 'Jalisco', 'Jiangsu', 'Johor', 'Limburg', 'New Territories', 'Ontario', 'Pahang', 'Pulau Pinang', 'Singapore', 'Texas'],
                    'Ship to country': ['USA', 'China', 'USA', 'Mexico', 'USA', 'Hong Kong', '', 'Mexico', 'China', 'Malaysia', 'Netherlands', 'Hong Kong', 'Canada', 'Malaysia', 'Malaysia', 'Singapore', 'USA']
                }

                sold_to_df = {
                    'Sold to Country/State': ['Ontario', 'Texas', 'Beijing', 'Jalisco', 'Pahang', 'Jiangsu', 'Internationl Country', 'New Territories', 'California', 'Limburg', 'Chihuahua', 'Singapore', 'Johor', 'Sao Paulo', 'Tamil Nadu'],
                    'Sold to country': ['Canada', 'USA', 'China', 'Mexico', 'Malaysia', 'China', '', 'Hong Kong', 'USA', 'Netherlands', 'Mexico', 'Singapore', 'Malaysia', 'Brazil', 'India']
                }

                

                ship_to_df = pd.DataFrame(ship_to_df)
                sold_to_df = pd.DataFrame(sold_to_df)
                df = pd.merge(df, ship_to_df, on='Ship to Country/State', how='left')
                df = pd.merge(df, sold_to_df, on='Sold to Country/State', how='left')
                df3 = df.groupby('Ship to country')['revenue'].sum()
                df4 = df.groupby('Sold to country')['revenue'].sum()
                




                if x == 'Sold to Country/State':

                    world_map = folium.Map(location=[0, 0], zoom_start=2)

                    # Define the coordinates and values for the heatmap
                    united_states = [37.0902, -95.7129,  df4.loc['USA']]  # latitude, longitude, value
                    china = [35.8617, 104.1954,  df4.loc['China']]  # latitude, longitude, value
                    brazil = [-14.2350, -51.9253,  df4.loc['Brazil']]  # latitude, longitude, value
                    canada = [56.1304, -106.3468,  df4.loc['Canada']]  # latitude, longitude, value
                    hong_kong = [22.3193, 114.1694,  df4.loc['Hong Kong']]  # latitude, longitude, value
                    india = [20.5937, 78.9629,  df4.loc['India']]  # latitude, longitude, value
                    malaysia = [4.2105, 101.9758,  df4.loc['Malaysia']]  # latitude, longitude, value
                    mexico = [23.6345, -102.5528,  df4.loc['Mexico']]  # latitude, longitude, value
                    netherlands = [52.1326, 5.2913,  df4.loc['Netherlands']]  # latitude, longitude, value
                    singapore = [1.3521, 103.8198,  df4.loc['Singapore']]  # latitude, longitude, value
                    data = [united_states, china, brazil, canada, hong_kong, india, malaysia, mexico, netherlands, singapore]

                    # Define the GeoJson layer and add it to the map
                    geojson_layer = folium.GeoJson(data={
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": None,
                                "properties": {
                                    "value": d[2]
                                }
                            } for d in data
                        ]
                    })
                    geojson_layer.add_to(world_map)

                    # Add the heatmap layer to the map
                    heatmap_layer = HeatMap(data=data, name='Heatmap', show=False)
                    heatmap_layer.add_to(world_map)

                    col1, col2 = st.columns([2, 7])
                    with col2:
                        folium_static(world_map)
                        
                        
                elif x == 'Ship to Country/State':
                    world_map = folium.Map(location=[0, 0], zoom_start=2)

                    # Define the coordinates and values for the heatmap

                    united_states = [37.0902, -95.7129, df3.loc['USA']]
                    china = [35.8617, 104.1954, df3.loc['China']]
                    canada = [56.1304, -106.3468, df3.loc['Canada']]
                    hong_kong = [22.3193, 114.1694, df3.loc['Hong Kong']]
                    malaysia = [4.2105, 101.9758, df3.loc['Malaysia']]
                    mexico = [23.6345, -102.5528, df3.loc['Mexico']]
                    netherlands = [52.1326, 5.2913, df3.loc['Netherlands']]
                    singapore = [1.3521, 103.8198, df3.loc['Singapore']]
                    data = [united_states, china, canada, hong_kong, malaysia, mexico, netherlands, singapore]

                    # Define the GeoJson layer and add it to the map
                    geojson_layer = folium.GeoJson(data={
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": None,
                                "properties": {
                                    "value": d[2]
                                }
                            } for d in data
                        ]
                    })
                    geojson_layer.add_to(world_map)

                    # Add the heatmap layer to the map
                    heatmap_layer = HeatMap(data=data, name='Heatmap', show=False)
                    heatmap_layer.add_to(world_map)

                    col1, col2 = st.columns([2, 7])
                    with col2:
                        folium_static(world_map)
                            

                
            
        
            if choice=="Manufacturer":
            
                st.write("Manufacture Leaderboard")
                # time_frame = st.sidebar.selectbox('Select timeframe to view KPIs',['Yearly','Quarterly','Monthly'],key=2)

                df_manu  = df[['Mock Manufacturer','Mock MPN','year','month','lead _time_in months','Order Qty','revenue','Remaining qty']]
                group_by = st.sidebar.selectbox('How do you want to group the data?',['Yearly','Monthly'], key=7)
                if group_by=='Yearly':
                    df_agg = df_manu.groupby(['Mock Manufacturer','Mock MPN','lead _time_in months','year']).agg(order_qty=('Order Qty', 'sum'), 
                               total_resale_amount=('revenue', 'sum'),open_qty = ('Remaining qty', 'sum')).reset_index()
                else:   
                    df_agg = df_manu.groupby(['Mock Manufacturer','Mock MPN','lead _time_in months','year','month']).agg(order_qty=('Order Qty', 'sum'), 
                               total_resale_amount=('revenue', 'sum'),open_qty = ('Remaining qty', 'sum')).reset_index()
                # df_agg_3 = df_agg_2[df_agg_2['year'].isin(year_val)]
                
                gb = GridOptionsBuilder.from_dataframe(df_agg)
                gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
                gb.configure_side_bar() #Add a sidebar
                gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
                gridOptions = gb.build()
                
                df_agg2 = df_agg.copy()
                df_agg2[['order_qty', 'total_resale_amount', 'open_qty']] = df_agg2[['order_qty', 'total_resale_amount', 'open_qty']].applymap('{:,.0f}'.format)
                grid_response = AgGrid(
                    df_agg2,
                    gridOptions=gridOptions,
                    data_return_mode='AS_INPUT', 
                    update_on='cellValueChanged', 
                    fit_columns_on_grid_load=False,
                    theme='alpine', #Add theme color to the table
                    enable_enterprise_modules=True,
                    height=400, 
                    width='100%',
                    reload_data=True
                )

                data = grid_response['data']
                selected = grid_response['selected_rows'] 
                
                df3 = pd.DataFrame(selected)
                if len(df3) ==1:
                    if group_by=='Yearly':
                        df_agg['time'] = df_agg['year']
                    else:
                        df_agg['time'] = df_agg['year'] + "-" + df_agg['month'] 
                    kpi = st.sidebar.selectbox('Select the Feature to view',['total_resale_amount','open_qty','order_qty'])
                    
                     
                    df_agg = df_agg[df_agg['Mock Manufacturer']==df3['Mock Manufacturer'].unique()[0]]
                    
                    time_frame = st.sidebar.multiselect("Select the time frame to view KPI's",df_agg['time'].unique(),default=df_agg['time'].unique()[-1])

                        
                    df_agg2 =df_agg[df_agg['time'].isin(time_frame)]
                    if kpi =='total_resale_amount':
                        col1.metric('Total Resale value','$ {:,} '.format(df_agg2[kpi].sum()))
                    else:
                        col1.metric('Total value','{:,} '.format(df_agg2[kpi].sum()))                    # fig = px.line(df_agg, x="lead _time_in months", y="sum_order_qty", title='Lead time vs Order QTY for %s'%(df3['Mock MPN'].unique()[0]))
                    chart_type = st.sidebar.selectbox('Select the Chart you want to view ',['MPNS with Lead time information','Historic Data'])
                    if chart_type=='MPNS with Lead time information':
                        st.write("MPN Level data for inividual manufacturer") 
                        st.plotly_chart(plotly_bar_chart(df=df_agg2,y_axis=kpi),use_container_width=True)
                    else:
                        st.write("Historic data for inividual manufacturer(NOT AFFECTED BY TIME FRAMES)") 
                        df_agg = df_agg.groupby(['Mock Manufacturer','time']).agg( total_resale_amount=('total_resale_amount', 'sum'),open_qty = ('open_qty', 'sum'),order_qty = ('order_qty', 'sum')).reset_index()
                        layout = go.Layout(
                        title="Historic chart value",
                        xaxis_title="time",
                        yaxis_title=kpi
                    )
                        time1 = df_agg['time'].values
                        kpi_val = df_agg[kpi].values
                        fig = go.Figure(
                            data=go.Scatter(x=time1, y=kpi_val),
                            layout=layout
                        )  
                        st.plotly_chart(fig,use_container_width=True)


            if choice=="Customer":
                st.write('Customer Leader board')
                df_cust = df[['Sold to Party Number',
                              'Sold to Country/State','Bill to Party Acct',	
                              'Bill to party Country/State',	'Ship to Party Number',
                                  	'Ship to Country/State','End Customer Number','Mock MPN','year','month','Order Qty','Unit Price','revenue','Remaining qty']]
                group_by = st.sidebar.selectbox('How do you want to group the data?',['Yearly','Monthly'], key=8)
                if group_by=='Yearly':

                    df_cust_agg = df_cust.groupby(['Sold to Party Number'	,
                              'Sold to Country/State','Bill to Party Acct',	
                              'Bill to party Country/State',	'Ship to Party Number',
                                  	'Ship to Country/State','End Customer Number','year']).agg(order_qty=('Order Qty', 'sum'), 
                               total_resale_amount=('revenue', 'sum'),open_qty = ('Remaining qty', 'sum')).reset_index()
                else:
                    df_cust_agg = df_cust.groupby(['Sold to Party Number'	,
                              'Sold to Country/State','Bill to Party Acct',	
                              'Bill to party Country/State',	'Ship to Party Number',
                                  	'Ship to Country/State','End Customer Number','year','month']).agg(order_qty=('Order Qty', 'sum'), 
                               total_resale_amount=('revenue', 'sum'),open_qty = ('Remaining qty', 'sum')).reset_index()

                gb = GridOptionsBuilder.from_dataframe(df_cust_agg)
                gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
                gb.configure_side_bar() #Add a sidebar
                gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
                gridOptions = gb.build()
                
                df_cust_agg2 = df_cust_agg.copy()
                df_cust_agg2[['order_qty', 'total_resale_amount', 'open_qty']] = df_cust_agg2[['order_qty', 'total_resale_amount', 'open_qty']].applymap('{:,.0f}'.format)

                grid_response = AgGrid(
                    df_cust_agg2,
                    gridOptions=gridOptions,
                    data_return_mode='AS_INPUT', 
                    update_on='cellValueChanged', 
                    fit_columns_on_grid_load=False,
                    theme='alpine', #Add theme color to the table
                    enable_enterprise_modules=True,
                    height=400, 
                    width='100%',
                    reload_data=True
                )

                data = grid_response['data']
                selected = grid_response['selected_rows'] 
                
                df3 = pd.DataFrame(selected)

                if len(df3)==1:
                    df_cust_agg = df_cust_agg[df_cust_agg['End Customer Number']==df3['End Customer Number'].unique()[0]]
                    if group_by=='Yearly':
                        df_cust_agg['time'] = df_cust_agg['year']
                    else:
                        df_cust_agg['time'] = df_cust_agg['year'] + "-" + df_cust_agg['month'] 

                    df_agg_cust_2 = df_cust_agg.groupby(['End Customer Number','time']).agg( total_resale_amount=('total_resale_amount', 'sum'),open_qty = ('open_qty', 'sum'),order_qty = ('order_qty', 'sum')).reset_index()
                    kpi = st.sidebar.selectbox('Select the Feature to view',['total_resale_amount','open_qty','order_qty'])
                    layout = go.Layout(
                        title="Historic chart value",
                        xaxis_title="time",
                        yaxis_title=kpi
                    )
                    time1 = df_agg_cust_2['time'].values
                    kpi_val = df_agg_cust_2[kpi].values
                    fig = go.Figure(
                        data=go.Scatter(x=time1, y=kpi_val),
                        layout=layout
                    )  
                    st.plotly_chart(fig,use_container_width=True)
                    time_frame = st.sidebar.multiselect("Select the time frame to view KPI's",df_cust_agg['time'].unique(),default=df_cust_agg['time'].unique()[-1])
                    df_cust_agg = df_cust_agg[df_cust_agg['time'].isin(time_frame)]
                    if kpi =='total_resale_amount':
                        col1.metric('Total Resale value','$ {:,} '.format(df_cust_agg[kpi].sum()))
                    else:
                        col1.metric('Total value','{:,} '.format(df_cust_agg[kpi].sum()))

    else:
        st.write('You are not authorised to view this page. Contact admin for access')
else:
    st.write('You are not logged in. Log in with your username and password in the home page')

