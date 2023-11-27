#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
import numpy as np
warnings.filterwarnings("ignore")


# In[2]:


df=yf.download("AAPL",period='5y')


# In[3]:


df=pd.DataFrame(df)


# In[4]:


df.head()


# In[5]:


df1=df[['Adj Close']]


# In[6]:


df1=df1['Adj Close'].resample('MS').mean()


# In[7]:


df1.head()


# In[8]:


df1.plot(figsize=(12, 5), legend=False)


# ##### Stationary Check

# In[9]:


import statsmodels.api as sm


# In[10]:


decomposition=sm.tsa.seasonal_decompose(df1,model='additive')
decomposition.plot().show()


# In[11]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df1, lags=50, ax=ax1)
plot_pacf(df1, lags=25, ax=ax2)
plt.show()


# In[12]:


from statsmodels.tsa.stattools import adfuller


# In[13]:


adftest=adfuller(df1)
print('pvalue of adfuller test is:', adftest[1])


# ##### Remove Stationarity

# In[14]:


diff_data=df1.diff().dropna()


# In[15]:


diff_data.plot()


# In[16]:


adftest2=adfuller(diff_data)
print('pvalue of adfuller test is:', adftest2[1])


# ##### Plot ACF and PACF 

# In[17]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(diff_data, lags=25, ax=ax1)
plot_pacf(diff_data, lags=25, ax=ax2)
plt.show()


# ##### Train Test Split

# In[18]:


len(df1)


# In[19]:


train=df1[:48]
test=df1[48:]


# ##### Find P, D, Q  Order Values 

# In[30]:


import itertools
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

p = range(0,5)
d = range(0,2)
q = range(0,5)

pdq_combination = list(itertools.product(p, d, q))

print('No of PDQ combinations: ', len(pdq_combination))

best_rmse = float('inf')
best_order = None

for pdq in pdq_combination:
    try:
        model = ARIMA(train, order=pdq).fit()
        pred = model.predict(start=len(train), end=(len(df1) - 1))
        error = np.sqrt(mean_squared_error(test, pred))
        if error < best_rmse:
            best_rmse = error
            best_order = pdq
    except:
        continue

print('Best PDQ order:', best_order)
print('Best RMSE:', best_rmse)


# In[21]:


import pmdarima as pm
auto_arima=pm.auto_arima(train,stepwise=False,seasonal=False)
auto_arima


# ##### Arima Model

# In[22]:


from statsmodels.tsa.arima.model import ARIMA


# In[26]:


model = ARIMA(train, order=(4,1,4)).fit()
pred=model.predict(start=len(train),end=(len(df1)-1))
error=np.sqrt(mean_squared_error(test,pred))
print(error)


# In[27]:


plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Testing Data')
plt.plot(test.index, pred, label='Predictions')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('ARIMA Model - Train, Test, and Predictions')
plt.show()


# ##### Predict Future Data

# In[28]:


#Training and predicting entire 5Years Data
final_model=ARIMA(df1,order=(4,1,4)).fit()
prediction=final_model.predict(start=len(df1),end=len(df1)+12)

#plot Graph
plt.plot(df1.index, df1, label='Training Data-5Y')
plt.plot(prediction.index, prediction, label='Predictions-1Y')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('ARIMA Model - Train and Predictions')
plt.show()

