import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression,LogisticRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split,KFold,GridSearchCV,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,r2_score,mean_squared_error,precision_recall_curve,f1_score,auc,roc_auc_score,roc_curve,make_scorer
from sklearn.externals import joblib

np.random.seed(9)   

train_data=pd.read_csv('Data/AdvWorksCusts.csv')
print('\t\t\tCustomer Spending Beheviour Prediction\n\n\t\tTraining Data\n',train_data.head(),'\n\n\t\tData Information\n')
train_data.info()

label_data1=pd.read_csv('Data/AW_AveMonthSpend.csv')
label_data2=pd.read_csv('Data/AW_BikeBuyer.csv')

train_data.drop_duplicates(subset='CustomerID',keep='last',inplace=True)
label_data1.drop_duplicates(subset='CustomerID',keep='last',inplace=True)
label_data2.drop_duplicates(subset='CustomerID',keep='last',inplace=True)

label1=label_data1['AveMonthSpend']
label2=label_data2['BikeBuyer']

train_data['age']=(train_data['BirthDate'].apply(lambda x:x[:4])).apply(lambda x:2019-np.int64(x))

train_data.drop(columns=['CustomerID','Title','FirstName','MiddleName','LastName','City','StateProvinceName','Suffix','AddressLine2','AddressLine1','PostalCode','PhoneNumber','BirthDate'],inplace=True)

num_col=['YearlyIncome','age']                     
cat_col=['CountryRegionName','Education','Occupation','Gender','MaritalStatus','HomeOwnerFlag','NumberCarsOwned','NumberChildrenAtHome','TotalChildren']

def vizual(data,columns,label=label2,type='bar'):
    if type=='bar':
        for col in columns:
            sns.countplot(y=data[col])
            plt.show()
    if type=='hist':
        for col in columns:
            sns.distplot(data[col],bins=30)
            plt.show()
    if type=='scatter':
        for col in columns:
            sns.scatterplot(col,label,data=data,s=10)
            plt.show()
    if type=='violin':
        for col in columns:
            if(data[col].dtype=='object'):
                sns.violinplot(col,label,data=data)
                plt.show()
            else:
                sns.violinplot(label,col,data=data)
                plt.show()  

print('\n\t\tData Vizualization\n\n\tBar Plot')     
vizual(train_data,cat_col,type='bar',label=label2[label2==1])
print('\tHistogram Plot')
vizual(train_data,num_col,type='hist',label=label2[label2==1])
print('\tScatter Plot')
vizual(train_data,num_col,type='scatter',label=label1)
print('\tViolin Plot')
vizual(train_data,train_data.columns,type='violin')

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

train_data=preperation(train_data)

print('\n\t\tData Vizualization After Data Modification\n\n\tHistogram Plot')     
vizual(train_data,['s_age'],type='hist')
print('\tScatter Plot')
vizual(train_data,['s_age'],type='scatter',label=label1)
print('\tBar Plot')     
vizual(train_data,['gnch','old'],type='bar')
print('\tViolin Plot')
vizual(train_data,['gnch','old'],type='violin')

def regression(x,y=label1,test_size=0.2):    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size)
    model=make_pipeline(PolynomialFeatures(degree=2),LinearRegression())
    model.fit(x_train,y_train)
    return model,x_test,y_test

def classify(x,y=label2,test_size=0.2):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size)
    model=make_pipeline(PolynomialFeatures(degree=2),LogisticRegression(C=0.7,penalty='l1',class_weight='balanced',solver='liblinear'))
    model.fit(x_train,y_train)
    return model,x_test,y_test

def roc_pr_f1(model,x,y,show=False,plot=False,show_all=False):
    no_skill_probs=[0 for i in range(len(y))]
    lr_probs=model.predict_proba(x)[:,1]
    pred=model.predict(x)
        
    ns_fpr,ns_tpr,_=roc_curve(y,no_skill_probs)
    lr_fpr,lr_tpr,_=roc_curve(y,lr_probs)
    
    ns_precision,ns_recall,_=precision_recall_curve(y,no_skill_probs)
    lr_precision,lr_recall,_=precision_recall_curve(y,lr_probs)
    ns_f1,ns_pr_auc,ns_roc_auc=0.0,auc(ns_recall,ns_precision),roc_auc_score(y,no_skill_probs)
    lr_f1,lr_pr_auc,lr_roc_auc=f1_score(y,pred),auc(lr_recall,lr_precision),roc_auc_score(y,lr_probs)
    
    if show or show_all:
        print('Accuracy score=',accuracy_score(y,pred))
        print('No Skill: F1 Score=',ns_f1,'\t\tPR AUC=',ns_pr_auc,' ROC AUC=',ns_roc_auc)
        print('Logistic: F1 Score=',lr_f1,' PR AUC=',lr_pr_auc,' ROC AUC=',lr_roc_auc)
        
    if plot or show_all:
        print('\n\tPrecision VS Recall')
        plt.plot(ns_recall,ns_precision,marker='.',label='No_Skill')
        plt.plot(lr_recall,lr_precision,marker='.',label='Logistics')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()
        
        print('\n\tROC Curve')
        plt.plot(ns_fpr,ns_tpr,marker='.',label='No_Skill')
        plt.plot(lr_fpr,lr_tpr,marker='.',label='Logistics')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend()
        plt.show()
    
    return lr_roc_auc,lr_pr_auc,lr_f1

def conf_mat(y_true,y_pred):
    cfm = confusion_matrix(y_true,y_pred)
    sns.heatmap(cfm, annot=True)
    print('\n\tConfusion Matrix')
    plt.xlabel('Predicted classes')
    plt.ylabel('Actual classes')  
    plt.show()

def evaluate(model,x_test,y_test,type='class',conf=True,show=False,plot=False,show_all=False):
    if type=='class':
        pred=model.predict(x_test)
        roc_auc,pr_auc,f1=roc_pr_f1(model,x_test,y_test,show,plot,show_all)
        if conf or show_all:
            conf_mat(y_test.values,pred)
    
        return (1-roc_auc)    
    
    if type=='reg':
        pred=model.predict(x_test)
        rmse=np.sqrt(mean_squared_error(y_test,pred))
        
        if show or show_all:
            print('R2 score = ',r2_score(y_test,pred),'\nRoot Mean Squared Error=',rmse) 
            
        if plot or show_all:
            error=pred-y_test
            plt.scatter(y_test,error,s=1)
            plt.xlabel('Actual value')
            plt.ylabel('Error')
            plt.show()
            
            error.hist(figsize=(5,5),bins=30)
            plt.xlabel('Error')
            plt.ylabel('Count')
            plt.show()
            
            plt.scatter(y_test,pred,s=1)
            plt.xlabel('Actual value')
            plt.ylabel('Predicted Value')
            plt.show()
            
        return rmse

def cv(model,data,label,scoring='roc_auc',fold=5):
    kfold=KFold(n_splits=fold,shuffle=True,random_state=6)
    score=cross_val_score(model,data,label,scoring=scoring,cv=kfold)
    return (score.mean())

def nested_cv(model,data,label,scoring='roc_auc',fold=10):
    inside=KFold(n_splits=fold,shuffle=True,random_state=2)
    outside=KFold(n_splits=fold,shuffle=True,random_state=7)
    clf=GridSearchCV(model,param_grid={"C":[0.1,0.7,1,10]},cv=inside,scoring=make_scorer(roc_auc_score),return_train_score=False)
    clf.fit(data,label)
    score=cross_val_score(clf,data,label,scoring=scoring,cv=outside)
    return clf.best_estimator_.C,(score.mean())

def predict(model,x_test,y_test):    
    predicted=pd.DataFrame()
    predicted['Actua Value']=y_test
    predicted['Predicted Value']=model.predict(x_test)
    return predicted

reg_model,x_test,y_test=regression(train_data)
print('\n\t\tPrediction on Validation set\n\tAvarage Monthly Speding of Customers\n',predict(reg_model,x_test,y_test),'\n\n\tEvaluation\n')
evaluate(reg_model,x_test,y_test,type='reg',show_all=True)
joblib.dump(reg_model,'Models/reg_model.sav')
cv_score=cv(reg_model,train_data,label=label1,scoring='r2')
print('\nMean R2 score after cross validation = ',cv_score)

class_model,x_test,y_test=classify(train_data,label2)
print('\n\tBike Buying Behaviour of Customers\n',predict(class_model,x_test,y_test),'\n\n\tEvaluation\n')
joblib.dump(class_model,'Models/class_model.sav')

evaluate(class_model,x_test,y_test,type='class',show_all=True)
cv_score=cv(class_model,train_data,label=label2,scoring='roc_auc')
print('\nMean ROC AUC after cross validation = ',cv_score)
