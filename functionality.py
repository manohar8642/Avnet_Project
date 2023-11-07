import pandas as pd
import numpy as np

from pmdarima.arima import auto_arima
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import logging
import modin.pandas as md
def initiate_logger():
    
    logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
      filename='/tmp/myapp.log',
      filemode='w')
    logging.info('logger Initiated')

def get_original_data():
    return pd.read_csv("Capstone Backlog Data.csv")

  

def model_define(dataframe,data_points,start_date,end_date,period_to_forecast,model):
    
    # If the number of data points is greater than 20, an ARIMA model will be run on the dataframe
    if data_points>15:
#         print("running exponential_smoothing model")
        # The dataframe is converted to have the first_of_month column as its index
        dataframe = dataframe.set_index('first_of_month')
        # The ARIMA model is defined in the arima function and is run on the dataframe with the specified start and end dates and the number of periods to forecast
        result,smape,rmse,str1 = ensemble_model(dataframe,start_date,end_date,period_to_forecast,model)
        # The exponential_smoothing model is defined in the arima function and is run on the dataframe with the specified start and end dates and the number of periods to forecast
#         exponential_smoothing(dataframe,start_date,end_date,period_to_forecast)

    # If the number of data points is greater than 10 but less than 20, a Moving Average model will be run on the dataframe
    elif data_points>10:
        print("running Moving Average model")
        
    # If the number of data points is less than or equal to 10, a Rule Based model will be run on the dataframe
    else:
        print("running Rule Based model")
    
    return result,smape,rmse, str1


def mpn_data(data_mpn, period_to_forecast,model):
    # select the data for a specific MPN
    # data_mpn = data_mpn[data_mpn['Mock MPN']==mpn]
    
    # convert the year and month columns to a datetime column
    data_mpn['first_of_month'] = pd.to_datetime(data_mpn[['year', 'month']].assign(day=1))
    
    # select the first of the month and Order Qty columns
    data_mpn = data_mpn[['first_of_month','Order Qty']]
    
    # print the MPN for which the data is being worked on
    # print("Working on MPN:", mpn)
    
    # get the start date of the data
    start_date = str(data_mpn['first_of_month'].min())
    # print(mpn + " started on " + start_date)
    
    # get the end date of the data
    end_date = str(data_mpn['first_of_month'].max())
    # print(mpn + " ended on " + end_date)
    
    # print("number of data points:", data_mpn.shape[0])
    
    # define the forecasting model based on the number of data points
    result, smape,rmse,str1 = model_define(data_mpn, data_mpn.shape[0], start_date, end_date, period_to_forecast,model)
    return result, smape,rmse,str1



def ensemble_model(dataframe, start_date, end_date, period_to_forecast,model):
    # Creates a date range object that represents the desired start and end dates with a frequency of "MS" (month start).
    date_range_tr = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Reindexes the dataframe using the created date range object and fills any missing values with 0.
    dataframe = dataframe.reindex(date_range_tr, fill_value=0)
    
    
    
    
    dataframe_sq = dataframe.apply(lambda x: np.sqrt(x))
    #dataframe_diff = dataframe_sq.diff().dropna()
    
    try:
        model_aa = pm.auto_arima(dataframe_sq[:-6], start_p=1, start_q=1, test='adf', max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=None, D=1, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
        forecast_aa = model_aa.predict(n_periods=6)
    except:
        model_aa = pm.auto_arima(dataframe_sq[:-6], start_p=1, start_q=1, test='adf', max_p=3, max_q=3, m=3, start_P=0, seasonal=False, d=None, D=1, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
        forecast_aa = model_aa.predict(n_periods=6)
    #last_actual = dataframe_diff.iloc[-7]
    #forecast_aa[0] = forecast_aa[0]+last_actual
    #for i in range(1,len(forecast_aa)):
    #    previous_forecast = forecast_aa[i-1]
    #    forecast_aa[i] = forecast_aa[i] + previous_forecast
    forecast_aa_squared = np.power(forecast_aa, 2)
    forecast_aa_last_6 = pd.DataFrame(forecast_aa_squared, columns=['Prediction'])
    
#     print(forecast_aa_last_6)
    
    actual = dataframe['Order Qty'].iloc[-6:].to_numpy()
    smape_aa = np.mean(200 * np.abs(actual - forecast_aa_squared) / (np.abs(actual) + np.abs(forecast_aa_squared))) * 1 / len(actual)
    print("SMAPE value for arima: ", smape_aa)
    std_aa = forecast_aa_last_6['Prediction'].std()
    forecast_aa_last_6['Prediction_L'] = forecast_aa_last_6['Prediction']-(1.645*(std_aa/np.sqrt(6)))
    forecast_aa_last_6['Prediction_U'] = forecast_aa_last_6['Prediction']+(1.645*(std_aa/np.sqrt(6)))
    #calculate RMSE
    rmse_aa = np.power(np.sum(np.power(actual-forecast_aa_squared,2)) * 1 / len(actual),0.5)

    
    
    
    
    # Fit the exponential smoothing model to the data
    last_6_points = dataframe['Order Qty'].iloc[-6:]
    model_es = ExponentialSmoothing(dataframe['Order Qty'].iloc[:-6], seasonal_periods=3,trend='add',seasonal='add').fit()
    forecast_es = model_es.forecast(6)
    forecast_es = np.maximum(forecast_es, 0)
    forecast_es_last_6 = pd.DataFrame(forecast_es, columns=['Prediction'])
#     print(forecast_es_last_6)
    std_es = forecast_es_last_6['Prediction'].std()
    forecast_es_last_6['Prediction_L'] = forecast_es_last_6['Prediction']-(1.645*(std_es/np.sqrt(6)))
    forecast_es_last_6['Prediction_U'] = forecast_es_last_6['Prediction']+(1.645*(std_es/np.sqrt(6)))
    # Calculate SMAPE
    actual = last_6_points.to_numpy()
    smape_es = np.mean(200 * np.abs(actual - forecast_es) / (np.abs(actual) + np.abs(forecast_es))) * 1 / len(actual)
    
    # Print the SMAPE value
    # print("SMAPE value for exponential smoothe: ", smape_es)
    # calculate rmse

    rmse_es = np.power(np.sum(np.power(actual-forecast_es,2)) * 1 / len(actual),0.5)

    # Prepare data for Prophet
    dataframe_p = dataframe.reset_index()
    dataframe_p = dataframe_p.rename(columns={"index": "ds", "Order Qty": "y"})

    # Define the hyperparameters
    hyperparameters = {
        'seasonality_mode': 'additive',
        'changepoint_prior_scale': 0.05,
        'holidays_prior_scale': 0.1,
        'yearly_seasonality': False,
        'weekly_seasonality': True,
        'daily_seasonality': True
    }
    
    

    # Fit the Prophet model to the data
    last_6_points = dataframe_p['y'].iloc[-6:]
    model_prophet = Prophet(**hyperparameters)
    model_prophet.fit(dataframe_p.iloc[:-6])

    # Make future predictions
    future = model_prophet.make_future_dataframe(periods=6, freq='MS', include_history=False)
    forecast_prophet_last_6 = model_prophet.predict(future)
    forecast_prophet_last_6 = forecast_prophet_last_6[['ds', 'yhat']].tail(6).reset_index(drop=True)
    forecast_prophet_last_6['yhat'] = np.maximum(forecast_prophet_last_6['yhat'], 0)

    # Calculate SMAPE
    actual = last_6_points.to_numpy()
    smape_prophet = np.mean(200 * np.abs(actual - forecast_prophet_last_6['yhat']) / (np.abs(actual) + np.abs(forecast_prophet_last_6['yhat']))) * 1 / len(actual)

    # Calculate prediction interval
    std_prophet = forecast_prophet_last_6['yhat'].std()
    forecast_prophet_last_6['Prediction_L'] = forecast_prophet_last_6['yhat'] - (1.645 * (std_prophet / np.sqrt(6)))
    forecast_prophet_last_6['Prediction_U'] = forecast_prophet_last_6['yhat'] + (1.645 * (std_prophet / np.sqrt(6)))
    forecast_prophet_last_6 = forecast_prophet_last_6.rename(columns={"yhat": "Prediction"})
    print(forecast_prophet_last_6)

    # Print the SMAPE value
    print("SMAPE value for Prophet model: ", smape_prophet)
    
    
    if model == "Arima":
        try:
            model = pm.auto_arima(dataframe_sq, start_p=1, start_q=1, test='adf', max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=None, D=1, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
            forecast = model.predict(n_periods=period_to_forecast)
        except:
            model = pm.auto_arima(dataframe_sq, start_p=1, start_q=1, test='adf', max_p=3, max_q=3, m=3, start_P=0, seasonal=False, d=None, D=1, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
            forecast = model.predict(n_periods=period_to_forecast)
        #lst_actual = dataframe_diff.iloc[-1]
        #forecast[0] = forecast[0]+lst_actual
        #for i in range(1,len(forecast)):
        #    previous_forecast = forecast[i-1]
        #    forecast[i] = forecast[i] + previous_forecast
        forecast_squared = np.power(forecast, 2)
        future_index = pd.date_range(start=dataframe.index[-1] + pd.DateOffset(months=1), periods=period_to_forecast, freq='MS')
        forecast = pd.DataFrame(forecast_squared,index = future_index,columns=['Prediction'])
        
        forecast['Prediction_L'] = forecast['Prediction']-(1.645*(std_aa/np.sqrt(6)))
        forecast['Prediction_U'] = forecast['Prediction']+(1.645*(std_aa/np.sqrt(6)))
        
        result = pd.concat([dataframe, forecast], axis=1)
        result = result.combine_first(forecast_aa_last_6)
#         print(result)
        smape = smape_aa
        rmse = rmse_aa
        if smape_es<smape_aa and smape_es<smape_prophet:
            str1 = "Better model available : Exponential smoothing"
        elif smape_prophet<smape_aa and smape_prophet<smape_es:
            str1 = "Better model available : Prophet"
        else:
            str1 =''
    elif model =='Prophet':
        try:
            model = Prophet(**hyperparameters)
            model.add_seasonality(name='monthly', period=30.5, fourier_order=10)
            model.add_seasonality(name='weekly', period=7, fourier_order=5)
            model.fit(dataframe_p)
        except:
            model = Prophet(**hyperparameters)
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.fit(dataframe_p)

        # Make future predictions
        future = model.make_future_dataframe(periods=period_to_forecast, freq='MS')
        forecast = model.predict(future)
        forecast = forecast[['ds', 'yhat']].tail(period_to_forecast).reset_index(drop=True)
        forecast['yhat'] = np.maximum(forecast['yhat'], 0)
        forecast = forecast.rename(columns={"yhat": "Prediction"})
        
        forecast['Prediction_L'] = forecast['Prediction']-(1.645*(std_prophet/np.sqrt(6)))
        forecast['Prediction_U'] = forecast['Prediction']+(1.645*(std_prophet/np.sqrt(6)))

        # Combine actuals with forecasted values
        result = pd.concat([dataframe_p.set_index('ds'), forecast.set_index('ds')], axis=1)
        result = result.combine_first(forecast_prophet_last_6.set_index('ds')).reset_index()
        result = result.rename(columns={"y": "Order Qty"})
        result = result.set_index("ds")
        smape = smape_prophet
        rmse = 0
        if smape_aa<smape_es and smape_aa<smape_prophet:
            str1 = "Better model available : Auto Arima"
        elif smape_es<smape_aa and smape_es<smape_prophet:
            str1 = "Better model available : Exponential Smoothing"
        else:
            str1 =''


        
    else:
        future_index = pd.date_range(start=dataframe.index[-1] + pd.DateOffset(months=1), periods=period_to_forecast, freq='MS')
        try:
            model = ExponentialSmoothing(dataframe['Order Qty'], seasonal_periods=12,trend='add',seasonal='add').fit()
        except:
            model = ExponentialSmoothing(dataframe['Order Qty'], seasonal_periods=6,trend='add',seasonal='add').fit()
        forecast = model.forecast(period_to_forecast)
        forecast = np.maximum(forecast, 0)
        forecast = pd.DataFrame(forecast,index = future_index,columns=['Prediction'])
        
        forecast['Prediction_L'] = forecast['Prediction']-(1.645*(std_es/np.sqrt(6)))
        forecast['Prediction_U'] = forecast['Prediction']+(1.645*(std_es/np.sqrt(6)))
        
        result = pd.concat([dataframe, forecast], axis=1)
        result = result.combine_first(forecast_es_last_6)
        rmse =rmse_es
        smape = smape_es
        if smape_aa<smape_es and smape_aa<smape_prophet:
            str1 = "Better model available : Auto Arima"
        elif smape_prophet<smape_aa and smape_es>smape_prophet:
            str1 = "Better model available : Prophet"
        else:
            str1 =''
#         print(result)
    return result,smape,rmse, str1
    
def revenue(data_mpn,mpn, time):
    # Filter data based on MPN
    
    # Filter data based on time period (year, month, or quarter)
    # data_mpn = df[df['Cust Required date'].dt.year == time]
    # data_mpn = df[df['Cust Required date'].dt.month == time]
    # data_mpn = df[df['Cust Required date'].dt.quarter == time]
    
    # Create new columns for year, month, quarter, and revenue
    data_mpn['year'] = data_mpn['Cust Required date'].dt.year
    data_mpn['month'] = data_mpn['Cust Required date'].dt.month
    data_mpn['quarter'] = data_mpn['Cust Required date'].dt.quarter
    data_mpn['revenue'] = data_mpn['Unit Price']*data_mpn['Order Qty']
    data_mpn['revenue'] = data_mpn['revenue'].round(0)
    
    # Select only the relevant columns for analysis
    data_mpn1 = data_mpn[['revenue','year','month','quarter']].copy()
    
    # Analyze data based on time period selected
    if time == 'Yearly':
        # Drop month and quarter columns since we are analyzing yearly data
        data_mpn1.drop(['month', 'quarter'], axis=1, inplace=True)
        # Group data by year and calculate total revenue and percentage change
        data_mpn1 = data_mpn1.groupby(['year']).sum()
        data_mpn1['percentage'] = data_mpn1['revenue'].pct_change()
        data_mpn1['percentage'] = data_mpn1['percentage'].apply(lambda x: f"{x:.2%}")
    elif time == 'Monthly':
        # Drop quarter column since we are analyzing monthly data
        data_mpn1.drop(['quarter'], axis=1, inplace=True)
        # Group data by year and month and calculate total revenue and percentage change
        data_mpn1 = data_mpn1.groupby(['year','month']).sum()
        data_mpn1['percentage'] = data_mpn1['revenue'].pct_change()
        data_mpn1['percentage'] = data_mpn1['percentage'].apply(lambda x: f"{x:.2%}")
    elif time == 'Quarterly':
        # Drop month column since we are analyzing quarterly data
        data_mpn1.drop(['month'], axis=1, inplace=True)
        # Group data by year and quarter and calculate total revenue and percentage change
        data_mpn1 = data_mpn1.groupby(['year','quarter']).sum()
        data_mpn1['percentage'] = data_mpn1['revenue'].pct_change()
        data_mpn1['percentage'] = data_mpn1['percentage'].apply(lambda x: f"{x:.2%}")
        
    # Return the analyzed data
    return data_mpn1['revenue'].iloc[-1],data_mpn1['percentage'].iloc[-1]
    
