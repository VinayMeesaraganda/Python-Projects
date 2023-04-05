#!/usr/bin/env python
# coding: utf-8

# In[41]:


#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#import csv file and displaying the data
df=pd.read_csv('telco-customer-churn.csv')
df


# In[3]:


#Filtering the data
df=df.drop(df.columns[0],axis=1).copy()


# In[4]:


#Printing information about the data file
df.info()


# In[5]:


#Looking for null values
df.isnull().sum()


# In[6]:


df1=df.copy()


# In[7]:


#Converting total charges from object to float data type
df1.TotalCharges=pd.to_numeric(df1.TotalCharges,errors='coerce')


# In[8]:


#reading the information about data
df1.info()


# In[9]:


df1.isnull().sum()


# In[10]:


df1=df1[~df1.TotalCharges.isnull()].copy()
df1


# In[11]:


#Finding Outliners
sns.set(style='whitegrid')
fig,ax=plt.subplots(figsize=(8,6))
b=sns.boxplot(data=df1[['SeniorCitizen','MonthlyCharges','tenure','Partner']])
plt.show()


# In[12]:


sns.boxplot(df1['TotalCharges'], orient='v')
plt.show()


# In[13]:


#Finding all the unique values of data
for i in df1:
    print(i)
    print(df[i].unique())


# In[14]:


#converting all the data to numeric
df1['gender'].replace(['Female','Male'],[0,1],inplace=True)
df1['Partner'].replace(['No','Yes'],[0,1],inplace=True)
df1['Dependents'].replace(['No','Yes'],[0,1],inplace=True)
df1['PhoneService'].replace(['No','Yes'],[0,1],inplace=True)
df1['MultipleLines'].replace(['No','Yes','No phone service'],[0,1,2],inplace=True)
df1['InternetService'].replace(['DSL','Fiber optic','No'],[0,1,2],inplace=True)
df1['OnlineSecurity'].replace(['No','Yes','No internet service'],[0,1,2],inplace=True)
df1['OnlineBackup'].replace(['No','Yes','No internet service'],[0,1,2],inplace=True)
df1['DeviceProtection'].replace(['No','Yes','No internet service'],[0,1,2],inplace=True)
df1['TechSupport'].replace(['No','Yes','No internet service'],[0,1,2],inplace=True)
df1['StreamingTV'].replace(['No','Yes','No internet service'],[0,1,2],inplace=True)
df1['StreamingMovies'].replace(['No','Yes','No internet service'],[0,1,2],inplace=True)
df1['PaperlessBilling'].replace(['No','Yes'],[0,1],inplace=True)
df1['Churn'].replace(['No','Yes'],[0,1],inplace=True)
df1['Contract'].replace(['Month-to-month','One year','Two year'],[0,1,2],inplace=True)
df1['PaymentMethod'].replace(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'],[0,1,2,3],inplace=True)


# In[15]:


df1


# In[16]:


#summary of statistics pertaining to the DataFrame columns
df1.describe().T


# In[17]:


df1.info()


# In[18]:


for i in df:
    plt.figure(figsize = (20,2))
    sns.countplot(x=df[i],hue='Churn',data=df)
    plt.show()


# In[ ]:


# sns.pairplot(data= df1,vars=['TotalCharges','MonthlyCharges','Contract','tenure'])


# In[19]:


plt.figure(figsize=(18,10))
annot=True
hm=sns.heatmap(df1.corr(),annot=annot)


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


#splitting the data set to 80:20 ratio
df1_train,df1_test=train_test_split(df1,test_size=0.2)


# In[22]:


df1_train


# In[23]:


df1_test


# In[24]:


import sweetviz as sv


# In[31]:


compare_report=sv.compare([df1_train,'Train'],[df1_test,'Test'])


# In[ ]:


compare_report.show_html()


# In[ ]:


churn_report=sv.compare(df1_train,df1_test,'Churn')


# In[ ]:


churn_report.show_html()


# ## Observations

# 1.The gender doesn't effect much churn
# 
# 2.The columns (Senior Citizen, Phone Service, Multiple Lines, Paperless Billing and Monthly Charges) have a positive churning rate
# 
# 3.The columns like Monthly Charges and Paperless Billing have the highest churning rate
# 
# 4.The columns like Partner, Dependents, Tenure, Internet Service, Online Security, Online Backup, Device Protection , Tech Support, Streaming Tv, Streaming Movies, Contract, Payment Method,Total Charges are having negative churning rate which are hepling the customers to stay in network
# 
# 5.Here tenure is having the lowest churn rate means the higher the tenure period the lesser the churning rate.

# # Project 2

# In[32]:


pip install imblearn


# In[33]:


pip install xgboost


# In[34]:


from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report,confusion_matrix
# import the class
from sklearn.linear_model import LogisticRegression
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
#Import XGBClassifier from xgboost
from xgboost import XGBClassifier
from sklearn.metrics import recall_score


# In[35]:


x=df1.iloc[:,:-1]
y=df1.Churn
x.head()


# In[36]:


y.head()


# In[37]:


#splitting the data into 80/20 ratio
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# ## 1.Naive Bias before smote

# In[38]:


gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
print(classification_report(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(data=cm,annot=True,fmt='g')
rs=recall_score(y_test, y_pred, average='weighted')
print('Recall score : {}'.format(rs) )


# ## 2.Logistic Regression before smote

# In[42]:


# instantiate the model (using the default parameters)
logreg = LogisticRegression()
# fit the model with data
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
print(classification_report(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(data=cm,annot=True,fmt='g')
rs=recall_score(y_test, y_pred, average='weighted')
a=print('Recall score : {}'.format(rs) )


# ## 3.Random Forest before smote

# In[43]:


rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
print(classification_report(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(data=cm,annot=True,fmt='g')
rs=recall_score(y_test, y_pred, average='weighted')
print('Recall score : {}'.format(rs) )


# ## 4. XGBoost before smote

# In[44]:


xgbc=XGBClassifier()
xgbc.fit(x_train,y_train)
y_pred=xgbc.predict(x_test)
print(classification_report(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(data=cm,annot=True,fmt='g')
rs=recall_score(y_test, y_pred, average='weighted')
print('Recall score : {}'.format(rs) )


# ## Applying SMOTE technique for imbalanced data

# In[45]:


from imblearn.over_sampling import SMOTE
smote =SMOTE()


# In[46]:


#applying the smote technique
x_train_smote,y_train_smote= smote.fit_resample(x_train.astype('float'),y_train)
x_test_smote,y_test_smote= smote.fit_resample(x_test.astype('float'),y_test)


# In[47]:


print(y_train.value_counts())
print(y_train_smote.value_counts())


# In[48]:


print(y_test.value_counts())
print(y_test_smote.value_counts())


# ## 1.Naive Bias after smote

# In[49]:


gnb.fit(x_train_smote,y_train_smote)
y_pred=gnb.predict(x_test_smote)
print(classification_report(y_test_smote,y_pred))
cm=confusion_matrix(y_test_smote,y_pred)
print(cm)
sns.heatmap(data=cm,annot=True,fmt='g')
rs=recall_score(y_test_smote, y_pred, average='weighted')
print('Recall score : {}'.format(rs) )


# ## 2.logistic regression before smote

# In[50]:


# fit the model with data
logreg.fit(x_train_smote,y_train_smote)
y_pred=logreg.predict(x_test_smote)
print(classification_report(y_test_smote,y_pred))
cm=confusion_matrix(y_test_smote,y_pred)
print(cm)
sns.heatmap(data=cm,annot=True,fmt='g')
rs=recall_score(y_test_smote, y_pred, average='weighted')
print('Recall score : {}'.format(rs) )


# ## 3.Random Forest after smote

# In[51]:


rfc.fit(x_train_smote,y_train_smote)
y_pred=rfc.predict(x_test_smote)
print(classification_report(y_test_smote,y_pred))
cm=confusion_matrix(y_test_smote,y_pred)
print(cm)
sns.heatmap(data=cm,annot=True,fmt='g')
rs=recall_score(y_test_smote, y_pred, average='weighted')
print('Recall score : {}'.format(rs) )


# ## 4. XGBoost after smote

# In[52]:


xgbc.fit(x_train_smote,y_train_smote)
y_pred=xgbc.predict(x_test_smote)
print(classification_report(y_test_smote,y_pred))
cm=confusion_matrix(y_test_smote,y_pred)
print(cm)
sns.heatmap(data=cm,annot=True,fmt='g')
rs=recall_score(y_test_smote, y_pred, average='weighted')
print('Recall score : {}'.format(rs) )


# ## Hyperparameter tuning for XGBoost

# In[53]:


from xgboost import XGBClassifier

XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,
       min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1)


# In[54]:


xgbc=XGBClassifier(max_depth=2,subsample=1,n_estimators=100,learning_rate=0.1,randoms_state=5)
xgbc.fit(x_train_smote,y_train_smote)
y_pred=xgbc.predict(x_test_smote)
print(classification_report(y_test_smote,y_pred))
cm=confusion_matrix(y_test_smote,y_pred)
print(cm)
sns.heatmap(data=cm,annot=True,fmt='g')
rs=recall_score(y_test_smote, y_pred, average='weighted')
print('Recall score : {}'.format(rs) )


# ## Hyperparameter tuning for RandomForest

# In[55]:


rfc.get_params()


# In[56]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
n_estimators_list = list(range(10,220,50))
criterion_list = ['gini', 'entropy']
max_depth_list = list(range(5,41,10))
max_depth_list.append(None)
min_samples_split_list = [x/1000 for x in list(range(5, 41, 10))]
min_samples_leaf_list = [x/1000 for x in list(range(5, 41, 10))]
max_features_list = ['sqrt', 'log2']

params_grid = {
    'n_estimators': n_estimators_list,
    'criterion': criterion_list,
    'max_depth': max_depth_list,
    'min_samples_split': min_samples_split_list,
    'min_samples_leaf': min_samples_leaf_list,
    'max_features': max_features_list
}

num_combinations = 1
for k in params_grid.keys(): num_combinations *= len(params_grid[k])

print('Number of combinations = ', num_combinations)
params_grid


# In[57]:


def my_roc_auc_score(model, X, y): return metrics.roc_auc_score(y, model.predict(X))
model_rf = RandomizedSearchCV(estimator=RandomForestClassifier(class_weight='balanced'),
                              param_distributions=params_grid,
                              n_iter=50,
                              cv=5,
                              scoring=my_roc_auc_score,
                              return_train_score=True,
                              verbose=2)


# In[58]:


model_rf.fit(x_train_smote,y_train_smote)
y_pred=model_rf.predict(x_test_smote)
print(classification_report(y_test_smote,y_pred))
cm=confusion_matrix(y_test_smote,y_pred)
print(cm)
sns.heatmap(data=cm,annot=True,fmt='g')
rs=recall_score(y_test_smote, y_pred, average='weighted')
print('Recall score : {}'.format(rs) )


# ## Observations
# 1. Before applying the smote technique logistic regression has the highest recall value i.e 0.8109452736318408
# 2. After applying the smote technique XGBoost has the highest recall value i.e 0.8539823008849557 and it is very close to Random forest classifier i.e 0.8416912487708947
# 3. After performing hypertuning for both Random Forest classifier and XGBoost has the highest recall rate i.e 0.86
# 4. Acoording to me after applying the smote technique the XGBoost is the best value with great recall and accuracy score 

# In[ ]:




