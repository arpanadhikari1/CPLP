import pandas as pd
import numpy as np
from sklearn.externals import joblib

test_data=pd.read_csv('Models/AW_test.csv')
test_data.drop_duplicates(subset='CustomerID',keep='last',inplace=True)
features=test_data.copy()
features['age']=(features['BirthDate'].apply(lambda x:x[-4:])).apply(lambda x:2019-np.int64(x))
features.drop(columns=['CustomerID','Title','FirstName','MiddleName','LastName','City','StateProvinceName','Suffix','AddressLine2','AddressLine1','PostalCode','PhoneNumber','BirthDate'],inplace=True)

def scale(data,columns,type='min_max'):
    for col in columns:
        if(type=='z score'):
            data[col]=(data[col]-np.mean(data[col]))/(np.std(data[col]))
        if(type=='min_max'):
            data[col]=(data[col]-np.min(data[col]))/(np.max(data[col])-np.min(data[col]))
        if(type=='log'):
            data[col]=np.log(data[col])
        if(type=='inverse'):
            data[col]=(1/data[col])
    return data       
   
def preperation(data):
    data['Gender'].replace({'M':0.9,'F':0.1},inplace=True)
    data['NumberChildrenAtHome'].replace({0:0.1,1:0.3,2:0.5,3:0.7,4:0.9,5:1.1},inplace=True)

    data['s_age']=data['Gender']*data['YearlyIncome']/data['age']*data['NumberChildrenAtHome']
    data['old']=data['age'].apply(lambda x:0 if x<=60 else 1)
    data['gnch']=data['Gender']*data['NumberChildrenAtHome']
    
    data['NumberChildrenAtHome'].replace({1:0,2:1,3:1,4:1,5:1},inplace=True)
    data['NumberCarsOwned'].replace({1:0,2:0,3:1,4:1},inplace=True)
    data['TotalChildren'].replace({1:0,2:0,3:1,4:1,5:1},inplace=True)
    
    data=pd.get_dummies(data)
    data=scale(data,columns=['s_age'],type='log')
    data=scale(data,data.select_dtypes('int64','float64'))
    
    return data

features=preperation(features)

reg_model=joblib.load('reg_model.sav')
test_data['Avarage Monthly Spending']=reg_model.predict(features)

class_model=joblib.load('class_model.sav')
test_data['Bike Buyer']=class_model.predict(features)

test_data.to_csv('Result.csv')
