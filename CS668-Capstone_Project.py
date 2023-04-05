#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries
import pandas as pd
import numpy as  np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#Read Test and Train Datasets
df1=pd.read_csv("traindemographics.csv")
df2=pd.read_csv("trainperf.csv")
df3=pd.read_csv("trainprevloans.csv")
df4=pd.read_csv("testdemographics.csv")
df5=pd.read_csv("testperf.csv")
df6=pd.read_csv("testprevloans.csv")
pd.set_option("max_columns",None)


# In[4]:


#Applying inner join on Train datasets 
Train1=pd.merge(df1,df3,how='inner',on='customerid')
Train=pd.merge(Train1,df2,how='inner',on='customerid')


# In[5]:


#Applying inner join on Test datasets 
Test2=pd.merge(df4,df6,how='inner',on='customerid')
Test=pd.merge(Test2,df5,how='inner',on='customerid')


# In[6]:


#Displaying Train dataset
Train.head(3)


# In[6]:


#displaying Test dataset
Test.head()


# In[7]:


#Displaying Dataset Info
Train.info()


# In[8]:


#Drop unwanted columns
Train=Train.drop(columns=['customerid','bank_branch_clients','longitude_gps','latitude_gps','creationdate_x','referredby_x','referredby_y','approveddate_y','creationdate_y','closeddate'])
Test=Test.drop(columns=['customerid','bank_branch_clients','longitude_gps','latitude_gps','creationdate_x','referredby_x','referredby_y','approveddate_y','creationdate_y','closeddate'])


# In[9]:


# Train[['firstduedate','firstrepaiddate','approveddate_x']] = Train[['firstduedate','firstrepaiddate','approveddate_x']].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S.%f')
# Test[['firstduedate','firstrepaiddate','approveddate_x']] = Test[['firstduedate','firstrepaiddate','approveddate_x']].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S.%f')


# In[10]:


# converting Birthdate to Age
from datetime import datetime, date
def age(born):
    born = datetime.strptime(born, "%Y-%m-%d %H:%M:%S.%f").date()
    today = date.today()
    return today.year - born.year - ((today.month,
                                        today.day) < (born.month,
                                                    born.day))

Train['Age'] = Train['birthdate'].apply(age)
Test['Age'] = Test['birthdate'].apply(age)


# In[11]:


#Replacing Age column with Birthdate
TrainAge=Train['Age'].values
Train=Train.drop(columns=['Age','birthdate'])
Train.insert(loc = 1,column = 'Age',value = TrainAge)
TestAge=Test['Age'].values
Test=Test.drop(columns=['Age','birthdate'])
Test.insert(loc = 1,column = 'Age',value = TestAge)


# In[15]:


# Checking for Null Values
Train.isnull().sum()


# In[16]:


#Filling null values
Train['employment_status_clients']=Train['employment_status_clients'].fillna('Unemployed')
Test['employment_status_clients']=Test['employment_status_clients'].fillna('Unemployed')


# In[17]:


#Filling null values
Train['level_of_education_clients']=Train['level_of_education_clients'].fillna('NoStudy')
Test['level_of_education_clients']=Test['level_of_education_clients'].fillna('NoStudy')


# In[18]:


# Checking for Null Values
Train.isnull().sum()


# In[19]:


#count number of duplicate rows
print("No of Duplicates in Train datset:",len(Train)-len(Train.drop_duplicates()))
print("No of Duplicates in Test datset:",len(Test)-len(Test.drop_duplicates()))


# In[20]:


# Drop Duplicates
Train=Train.drop_duplicates()


# In[21]:


#pip install openpyxl


# In[22]:


# Export data set to local system for data Analysis in Tableau
#Train.to_excel(r'C:\Users\rajvi\OneDrive\Desktop\Capstone Project\Data Sets\FinalTrain.xlsx', index=False)


# In[14]:


#Converting Categorical to Numerical Values
Train['good_bad_flag'].replace(['Good', 'Bad'],[1,0], inplace=True)


# In[24]:


#Converting Categorical to Numerical Values
Train['bank_account_type'].replace(['Savings', 'Current','Other'],[1,2,3], inplace=True)
Test['bank_account_type'].replace(['Savings', 'Current','Other'],[1,2,3], inplace=True)


# In[25]:


#Converting Categorical to Numerical Values
Train['employment_status_clients'].replace(['Permanent', 'Self-Employed','Unemployed','Student','Retired','Contract'],[1,2,3,4,5,6], inplace=True)
Test['employment_status_clients'].replace(['Permanent', 'Self-Employed','Unemployed','Student','Retired','Contract'],[1,2,3,4,5,6], inplace=True)


# In[26]:


#Converting Categorical to Numerical Values
Train['level_of_education_clients'].replace(['NoStudy', 'Graduate','Secondary','Post-Graduate','Primary'],[0,3,2,4,1], inplace=True)
Test['level_of_education_clients'].replace(['NoStudy', 'Graduate','Secondary','Post-Graduate','Primary'],[0,3,2,4,1], inplace=True)


# In[27]:


#import label encoder
from sklearn import preprocessing 
#make an instance of Label Encoder
label_encoder = preprocessing.LabelEncoder()
#Converting Categorical to Numerical Values using Label Encoder
Train["bank_name_clients"] = label_encoder.fit_transform(Train["bank_name_clients"])
Test["bank_name_clients"] = label_encoder.fit_transform(Test["bank_name_clients"])


# In[28]:


# Plot Heat Map using seaborn
plt.figure(figsize=(18,10))
annot=True
hm=sns.heatmap(Train.corr(),annot=annot)


# ## Machine learning Models

# ### Random Forest Classifier

# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


# In[30]:


x=Train.iloc[:,:-1]
y=Train.good_bad_flag


# In[31]:


# Splitting Train dataset into into 80:20 ratio
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[32]:


# List of  Features for Training ML Models
features=["Age","bank_account_type","bank_name_clients","employment_status_clients","level_of_education_clients","systemloanid_x","loannumber_x","loanamount_x","totaldue_x","termdays_x","systemloanid_y","loannumber_y","loanamount_y","totaldue_y","termdays_y"]


# In[33]:


# Target Feature in ML Model
target=["good_bad_flag"]


# In[34]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42,criterion='gini',max_features='sqrt')


# In[35]:


#fit and predict the data
rf.fit(x_train[features],y_train)
y_pred=rf.predict(x_test[features])


# In[36]:


#Created lists to save scores
Accuracy_score=[]
AUC_score=[]
MAE_Score=[]
name=[]


# In[37]:


MAE_Rf = round(mean_absolute_error(y_pred,y_test)*100,2)
AUC_score_Rf = round(roc_auc_score(y_test,y_pred)*100,2)
Accuracy_Rf=round(accuracy_score(y_test,y_pred)*100,2)
print("Accuracy :",Accuracy_Rf)
print("Mean absolute error:",MAE_Rf)
print("AUC_Score :", AUC_score_Rf)
Accuracy_score.append(Accuracy_Rf)
AUC_score.append(AUC_score_Rf)
MAE_Score.append(MAE_Rf)
name.append("Random Forest")


# ### Logistic Regression

# In[38]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(fit_intercept=True,max_iter=100,n_jobs=1,solver='liblinear')


# In[39]:


#fit and predict the data
lr.fit(x_train[features],y_train)
y_pred=lr.predict(x_test[features])


# In[40]:


#Performance Display 
MAE_Lr = round(mean_absolute_error(y_pred,y_test)*100,2)
AUC_score_Lr = round(roc_auc_score(y_test,y_pred)*100,2)
Accuracy_Lr=round(accuracy_score(y_test,y_pred)*100,2)
print("Accuracy :",Accuracy_Lr)
print("Mean absolute error:",MAE_Lr)
print("AUC_Score :", AUC_score_Lr)
Accuracy_score.append(Accuracy_Lr)
AUC_score.append(AUC_score_Lr)
MAE_Score.append(MAE_Lr)
name.append("Logistic Regression")


# ### XGBOOST

# In[41]:


from xgboost import XGBClassifier
XG=XGBClassifier(eval_metric='logloss')


# In[42]:


#fit and predict the data
XG.fit(x_train[features],y_train)
y_pred=XG.predict(x_test[features])


# In[43]:


#Performance Display 
MAE_XG = round(mean_absolute_error(y_pred,y_test)*100,2)
AUC_score_XG = round(roc_auc_score(y_test,y_pred)*100,2)
Accuracy_XG=round(accuracy_score(y_test,y_pred)*100,2)
print("Accuracy :",Accuracy_XG)
print("Mean absolute error:",MAE_XG)
print("AUC_Score :", AUC_score_XG)
Accuracy_score.append(Accuracy_XG)
AUC_score.append(AUC_score_XG)
MAE_Score.append(MAE_XG)
name.append("XG Boost")


# ### Decision Tree

# In[45]:


from sklearn.tree import DecisionTreeClassifier
Desc_Tree=DecisionTreeClassifier()


# In[46]:


#fit and predict the data
Desc_Tree.fit(x_train[features],y_train)
y_pred = Desc_Tree.predict(x_test[features])


# In[47]:


#Performance Display 
MAE_DT = round(mean_absolute_error(y_pred,y_test)*100,2)
AUC_score_DT = round(roc_auc_score(y_test,y_pred)*100,2)
Accuracy_DT=round(accuracy_score(y_test,y_pred)*100,2)
print("Accuracy :",Accuracy_DT)
print("Mean absolute error:",MAE_DT)
print("AUC_Score :", AUC_score_DT)
Accuracy_score.append(Accuracy_DT)
AUC_score.append(AUC_score_DT)
MAE_Score.append(MAE_DT)
name.append("Decision Tree")


# In[48]:


#Comparision of Machine Learning Models
scores = pd.DataFrame({'name': name,'Accuracy_score': Accuracy_score,'AUC_score': AUC_score,'MAE_Score':MAE_Score})
scores


# In[ ]:




