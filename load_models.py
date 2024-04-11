#!/usr/bin/env python
# coding: utf-8

# ## Importation

# In[8]:


import pickle
import pandas as pd # data science library to manipulate data
import numpy as np # mathematical library to manipulate arrays and matrices
import matplotlib.pyplot as plt # visualization library
from sklearn.feature_selection import SelectKBest # selection method
from sklearn.feature_selection import mutual_info_regression,f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import  linear_model

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import  metrics
from sklearn.neural_network import MLPRegressor


# In[9]:


# regions=['Auvergne-Rhône-Alpes']
regions=['Auvergne-Rhône-Alpes','Bourgogne-Franche-Comté','Bretagne','Centre-Val-de-Loire','Grand-Est','Hauts-de-France','Ile-de-France','Normandie','Nouvelle-Aquitaine','Occitanie','PACA','Pays-de-la-Loire']


# ## Load Data and Models

# In[10]:


with open('data/power&meteo.pkl','rb') as file:
    (df_power,df_meteo,df_merged)=pickle.load(file)

for region in regions:
    # df_merged[region]=df_merged[region].drop_duplicates(subset=['Consommation-1'])
    df_consommation_1 = df_merged[region]['Consommation-1'].copy()

    df_merged[region] = df_merged[region].drop(columns=['Consommation-1'])

    df_merged[region]['Consommation-1'] = df_consommation_1.iloc[:, [0]]

for region in regions:
    df_meteo[region]['Hour'] = df_meteo[region].index.hour
    df_power[region].replace({'-': 0}, inplace=True)


# In[11]:


ss_Y_cons={}
ss_X_cons={}
features_names_cons={}
model_cons={}

for region in regions:
    with open('models/'+region+'RFcons.pkl','rb') as file:
        model_cons[region],ss_Y_cons[region], ss_X_cons[region],features_names_cons[region]=pickle.load(file)


# In[12]:


ss_Y_prod={}
ss_X_prod={}
features_names_prod={}
model_prod={}

for region in regions:
    with open('models/'+region+'RFprod.pkl','rb') as file:
        model_prod[region],ss_Y_prod[region], ss_X_prod[region],features_names_prod[region]=pickle.load(file)


# In[13]:


# print(features_names_prod[region])


# ## Load Model and use it

# In[14]:


# regions=['Auvergne-Rhône-Alpes']
regions=['Auvergne-Rhône-Alpes','Bourgogne-Franche-Comté','Bretagne','Centre-Val-de-Loire','Grand-Est','Hauts-de-France','Ile-de-France','Normandie','Nouvelle-Aquitaine','Occitanie','PACA','Pays-de-la-Loire']
date_test='2022-07-19 00:00:00'

def predict_next_hour_cons(region,
                      date,
                      df_met=df_meteo,
                      df_pow=df_power,
                      power_input={'Consommation-1':df_power[region].loc[date_test, 'Consommation']}):
    power=power_input
    date = pd.to_datetime(date)
    date1 = date + pd.Timedelta(hours=3)
    X_now=[]
    k=0
    power['list_pow']=[]
    for feature in features_names_cons[region]:
        if feature in df_met[region].columns :
            X_now.append(df_met[region].loc[date1,feature])
        # elif feature[:-2] in df_pow[region].columns :
        #     X_now.append(df_pow[region].loc[date,feature[:-2]])
        else:
            X_now.append(power[feature])
            power['list_pow'].append([feature,k])
        k=k+1
    
    X_now=np.array([[float(k) for k in X_now]])
    

   
    y_pred_ss = model_cons[region][0].predict(ss_X_cons[region].transform(X_now))

    y_pred=ss_Y_cons[region].inverse_transform(y_pred_ss.reshape(-1,1))
    for i in np.array(power['list_pow']):
        # name_feature = i[0]
        # print(i[0][1])
        power[i[0]] = y_pred[0][0]#[i[1]]
    

    
    return y_pred[0,0]
    
# regions=['Auvergne-Rhône-Alpes']#,'Bourgogne-Franche-Comté','Bretagne','Centre-Val-de-Loire','Grand-Est','Hauts-de-France','Ile-de-France','Normandie','Nouvelle-Aquitaine','Occitanie','PACA','Pays-de-la-Loire']

# date_test='2022-07-19 00:00:00'
# for region in regions:
#     print(predict_next_hour_cons(region,date=date_test,power_input={'Consommation-1':df_power[region].loc[date_test, 'Consommation']}))



def predict_next_day_cons(region,
                    date,
                    df_met=df_meteo,
                    df_pow=df_power):
    
    date = pd.to_datetime(date)
    date1 = date + pd.Timedelta(hours=3)
    date1_end = date1 + pd.Timedelta(days=1)

    next_day = pd.date_range(start=date, end=date1_end, freq='3H')

    Yp=[df_pow[region].loc[date, 'Consommation']]
    real_power=[]
    for current_date in next_day:
        power=Yp[-1]  
        real_power.append(df_pow[region].loc[current_date, 'Consommation']) 
        # prev = predict_next_hour_consbof(region=region,date=current_date)#,power=power)

        prev=predict_next_hour_cons(region=region,date=current_date,power_input={'Consommation-1':power})
        Yp=np.append(Yp, prev)
            
    return [float(k) for k in Yp[:-1]],next_day#,[float(k) for k in real_power]

def predict_next_day_value_cons(region,
                    date,
                    df_met=df_meteo,
                    df_pow=df_power):
    Y,X=predict_next_day_cons(region,date,df_met,df_pow)
    energy=0
    for y in Y:
        energy+=y*3
    return energy



# date_test='2022-07-19 00:00:00'
# regions=['Ile-de-France']
# for region in regions:
#     # print(predict_next_day_cons(region,date_test))

#     plt.plot(predict_next_day_cons(region,date_test)[1], predict_next_day_cons(region,date_test)[0])

#     plt.plot(predict_next_day_cons(region,date_test)[1], predict_next_day_cons(region,date_test)[2])
    
#     plt.xlabel('Dates')
#     plt.ylabel('Power_prevision_'+region)
#     plt.show()


def predict_next_hour_prod(region,date,power_input,
                            df_met=df_meteo,
                            df_pow=df_power):
    power=power_input
    date = pd.to_datetime(date)
    date1 = date + pd.Timedelta(hours=3)
    X_now=[]
    # k=0
    power['list_pow']=[]
    for feature in features_names_prod[region]:
        k=0
        if feature in df_met[region].columns :
            X_now.append(df_met[region].loc[date1,feature])
        # elif feature[:-2] in df_pow[region].columns :
        #     X_now.append(df_pow[region].loc[date,feature[:-2]])
        else:
            X_now.append(power[feature])
            power['list_pow'].append([feature,k])
        k=k+1
    
    X_now=np.array([[float(k) for k in X_now]])
    

   
    y_pred_ss = model_prod[region][0].predict(ss_X_prod[region].transform(X_now))

    y_pred=ss_Y_prod[region].inverse_transform(y_pred_ss)
    for i in np.array(power['list_pow']):
        power[i[0]] = y_pred[0][int(i[1])]
    
    return y_pred[0],power
    
# regions=['Auvergne-Rhône-Alpes']#,'Bourgogne-Franche-Comté','Bretagne','Centre-Val-de-Loire','Grand-Est','Hauts-de-France','Ile-de-France','Normandie','Nouvelle-Aquitaine','Occitanie','PACA','Pays-de-la-Loire']

power_inputt={'Thermique-1':df_power[region].loc[date_test, 'Thermique'],
             'Nucléaire-1':df_power[region].loc[date_test, 'Nucléaire'],
             'Eolien-1':df_power[region].loc[date_test, 'Eolien'],
             'Solaire-1':df_power[region].loc[date_test, 'Solaire'],
             'Hydraulique-1':df_power[region].loc[date_test, 'Hydraulique'],
             'Bioénergies-1':df_power[region].loc[date_test, 'Bioénergies'],
             'Pompage-1':df_power[region].loc[date_test, 'Pompage']}
date_test='2022-07-19 00:00:00'
# for region in regions:
#     print(predict_next_hour_prod(region,date=date_test,power_input=power_inputt)[0])


# In[21]:


def predict_next_day_prod(region,
                    date,
                    df_met=df_meteo,
                    df_pow=df_power):
    
    date = pd.to_datetime(date)
    date1 = date + pd.Timedelta(hours=3)
    date1_end = date1 + pd.Timedelta(days=6)
    real_power=[]

    next_day = pd.date_range(start=date, end=date1_end, freq='3H')
    power={'Thermique-1':df_pow[region].loc[date, 'Thermique'],
             'Nucléaire-1':df_pow[region].loc[date, 'Nucléaire'],
             'Eolien-1':df_pow[region].loc[date, 'Eolien'],
             'Solaire-1':df_pow[region].loc[date, 'Solaire'],
             'Hydraulique-1':df_pow[region].loc[date, 'Hydraulique'],
             'Bioénergies-1':df_pow[region].loc[date, 'Bioénergies'],
             'Pompage-1':df_pow[region].loc[date, 'Pompage']}

    Yp=[[df_pow[region].loc[date, k] for k in ['Thermique','Nucléaire','Eolien','Solaire','Hydraulique','Pompage','Bioénergies']]]
    
    for current_date in next_day:
        prev,power=predict_next_hour_prod(region,date=current_date,power_input=power,df_met=df_meteo,df_pow=df_power)
        
        Yp.append(prev)
        real_power.append([df_pow[region].loc[current_date, k] for k in ['Thermique','Nucléaire','Eolien','Solaire','Hydraulique','Pompage','Bioénergies']])
    Yp=np.array(Yp)
    for k in range(Yp.shape[0]):
        for i in range(Yp.shape[1]):
            Yp[k][i]=float(Yp[k,i]) 

    real_power=np.array(real_power)

    for k in range(real_power.shape[0]):
        for i in range(real_power.shape[1]):
            real_power[k][i]=float(real_power[k,i])   
    
    return Yp[:-1, :],next_day,np.array(real_power)


# date_test='2022-07-19 00:00:00'
# regions=['Ile-de-France']
# for region in regions:
#     # print(predict_next_day_prod(region,date_test))
#     for j in range(predict_next_day_prod(region,date_test)[0].shape[1]):
#         j=6

#         plt.plot(predict_next_day_prod(region,date_test)[1], predict_next_day_prod(region,date_test)[0][:, j],label=['Thermique','Nucléaire','Eolien','Solaire','Hydraulique','Pompage','Bioénergies'][j])
    
#         plt.plot(predict_next_day_prod(region,date_test)[1], predict_next_day_prod(region,date_test)[2][:, j],label=['Thermique','Nucléaire','Eolien','Solaire','Hydraulique','Pompage','Bioénergies'][j]+'real')
#         break
#     plt.xlabel('Dates')
#     plt.ylabel('Power_prevision_'+region)
#     plt.legend()
#     plt.show()


# In[ ]:




