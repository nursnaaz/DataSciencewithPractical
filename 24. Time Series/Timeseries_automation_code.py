# -*- coding: utf-8 -*-
"""
Created on Mon Dec  19 14:44:06 2019

@author: Mohamed.Imran
"""

import os
import pandas as pd
import numpy as np
import datetime as dt
from sklearn import linear_model
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from fbprophet import Prophet
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

os.chdir(r"C:\Users\moham\OneDrive - University of Central Missouri\Desktop\Imran\Training\Inceptez\Content\Data-Science-Central-Online-master\Data-Science-Central-Online-master\Time Series")
pwd = os.getcwd()

data = pd.read_excel("TS_Data.csv") #Read your decomposition file here

#Regression model
def Regression(data):
    data["Date"] = pd.to_datetime(data["Date"])
    data=data.sort_values(["Date"])
    data["Quarter"] = data["Date"].dt.quarter
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    
    Train=data[(data.Year>=2015)&(data.Year<2019)] #modify date according to your dataset; Train : 2017-2018
    Test=data[(data.Year==2019)]  #modify date according to your dataset; Test : 2019
    
    
    Train["SI_Y"]=Train["Volume"]/Train.groupby("Year")["Volume"].transform(np.mean)
    Train["F_SI"]=Train.groupby("Month")["SI_Y"].transform(np.mean)
    Train["D_Seasonalised_trend"] = Train["Volume"]/Train["F_SI"]    
    Train["Level_index1"]=np.mean(Train[(Train.Year==2018)&(Train.Quarter==1)]["D_Seasonalised_trend"])/np.mean(Train[(Train.Year==2017)&(Train.Quarter==4)]["D_Seasonalised_trend"])
    
    numer1=np.mean(Train[(Train.Year==2018)&(Train.Quarter==3)]["D_Seasonalised_trend"])/np.mean(Train[(Train.Year==2018)&(Train.Quarter==2)]["D_Seasonalised_trend"])
    numer2=np.mean(Train[(Train.Year==2018)&(Train.Quarter==4)]["D_Seasonalised_trend"])/np.mean(Train[(Train.Year==2018)&(Train.Quarter==3)]["D_Seasonalised_trend"])
    
    
    Train["Level_index2"]=np.mean([numer1,numer2])
    Train=Train.sort_values(["Date"])
    Train.index=range(len(Train))
    Train["ID"]=range(1,(len(Train)+1))
    
    Train["Deleveled_series"]=np.where(Train.Year==2017, Train["D_Seasonalised_trend"]*Train["Level_index1"],Train["D_Seasonalised_trend"])
    
    lm = linear_model.LinearRegression()
    X = np.array(Train[["ID", "Variable_1"]]) # In case of no extra variable in the dataset, remove the extra variable name from the list, then append the line with ".reshape(-1, 1)"
    Y = np.array(Train["Deleveled_series"]).reshape(-1,1)
    
    model = lm.fit(X,Y)
    
    Test["ID"]=range(len(Test))
    Test["ID"]=Test["ID"]+max(Train["ID"])
    X_test=np.array(Test[["ID", "Variable_1"]]) # In case of no extra variable in the dataset, remove the extra variable name from the list, then append the line with ".reshape(-1, 1)"
    Y_test=model.predict(X_test)
    
    Pred1 = Y_test*Train.iloc[0]["Level_index2"]*np.array(Train.iloc[0:len(Y_test)]["F_SI"]).reshape(-1,1)
    Test["Predictions"]=Pred1
    
    return(Test['Predictions'])

#Arima model
def Arima(data): 
    X = data['Volume'].values
    size = np.sum(data['Date']<='12/31/2018')
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()  
        
    for t in range(len(test)):
    	model = ARIMA(history, order=(1,1,0))
    	model_fit = model.fit(disp=0)
    	output = model_fit.forecast()
    	yhat = output[0]
    	predictions.append(yhat)
    	obs = test[t]
    	history.append(obs)
    return predictions   

#Holts-Winter model
def Holts_winter(data):
    inter_df = data[['Volume']]
    size = np.sum(data['Date']<='12/31/2018')
    train, test = inter_df.iloc[:size, 0], inter_df.iloc[size:, 0]
    model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=12).fit()
    pred = model.predict(start=test.index[0], end=test.index[-1])
    return pred

#Fbprophet
def Fbprophet(data):
    size = np.sum(data['ds']<='12/31/2018')
    inter_df = data.iloc[:size, :]
    m = Prophet(weekly_seasonality=False, daily_seasonality=False)
    m.fit(inter_df)
    future = m.make_future_dataframe(periods=12, freq='M')
    forecast = m.predict(future)
    fcst = forecast['yhat'].tail(12)
    return fcst

#Simple Exponential Smoothing model
def Ses(data):
    inter_df = data[['Volume']]
    size = np.sum(data['Date']<='12/31/2018')
    train, test = inter_df.iloc[:size, 0], inter_df.iloc[size:, 0]
    model = SimpleExpSmoothing(train).fit()
    pred = model.predict(start=test.index[0], end=test.index[-1])
    return pred



def Regression_2lag(data):
    data["Variable_1"] = data["Variable_1"].shift(2)
    data = data.loc[2:, :]
    return Regression(data)


required_cols = [col for col in data.columns if col not in ['Date', 'Variable_1']]


Result=pd.DataFrame()

for model in [Regression, Arima, Holts_winter, Ses, Fbprophet, Regression_2lag]:
    for i in required_cols:
        data['Date'] = pd.to_datetime(data['Date'])
        to_func = data[["Date", "Variable_1", i]]
        to_func.columns=["Date","Variable_1", "Volume"]
        if model == Fbprophet:
            to_func.columns=["ds","Variable_1", "y"]
            Result_inter = model(to_func[['ds', 'y']])
            Result_inter.name = model.__name__ + "_" +  i
            Result_inter.index = range(len(Result_inter))
            Result = pd.concat([Result, Result_inter], axis = 1)
        elif model == Arima:
            Result_inter = model(to_func)
            Result_inter = pd.DataFrame(Result_inter, columns = ["ARIMA_" + i])
            Result_inter.index=range(len(Result_inter))
            Result = pd.concat([Result, Result_inter], axis = 1)
        else:
            Result_inter = model(to_func)
            Result_inter.name = model.__name__ + "_" +  i
            Result_inter.index=range(len(Result_inter))
            Result = pd.concat([Result, Result_inter], axis = 1)



Result.to_csv('Forecast.csv')

