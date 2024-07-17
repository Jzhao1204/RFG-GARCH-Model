#!/usr/bin/env python
# coding: utf-8

# In[16]:


#Statistical Properties
import yfinance as yf
data= yf.download("RFG", start="2000-01-01", end="2024-05-13")
data['Daily Returns'] = 100 * data['Adj Close'].pct_change()
print(data['Daily Returns'].describe())

#skewness
print('RFG Skewness:', data['Daily Returns'].skew())

#kurtosis
print('RFG Kurtosis:', data['Daily Returns'].kurtosis())

#Discussion: Mean is slightly off from 0 at 0.0475.
#Distribution is skewed -0.5172 to the left.
#RFG has fat tails at 8.7856 as it is greater than the normal at 3.


# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

data['abs_returns']=np.abs(data['Daily Returns'])
data_2=data.dropna()
mean=data_2['abs_returns'].mean()
plt.plot(data_2['abs_returns'])
plt.axhline(y=mean,color='red',label='Average Absolute Return')
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('RFG Absolute Returns')
plt.show()
#Discussion: Volatility clustering is apparent in the graph as large changes in volatility are followed by another large change.
#Periods of high and low volatility are observable over time and provides evidence of time-varying volatility.


# In[18]:


sm.graphics.tsa.plot_acf(data_2['Daily Returns'],lags=30,zero=False,auto_ylims=True)
plt.title('ACF of Returns for RFG')
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.show()

sm.graphics.tsa.plot_acf(data_2['abs_returns'],lags=30,zero=False,auto_ylims=True)
plt.title('ACF of Absolute Returns for RFG')
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.show()

#Discussion: There are 9 instances of violations and points outside of the threshold level for the returns for RFG.


# In[19]:


#Modeling Volatility with EMA
data_2['EMA']=(data_2['Daily Returns']**2).ewm(alpha=0.06).mean()**0.5
data_2['EMA'].plot(kind='line')
#Discussion: There are 3 instances where the returns are high.
#However, after the window is complete, the impact of these high values are no longer in effect.


# In[20]:


#GARCH Modeling
from arch import arch_model
am= arch_model(data_2['Daily Returns'])
res=am.fit()
print(res)

#Discussion: The coefficient estimates of alpha and beta are 0.0939 and 0.8911 respectively.
#The sum of alpha and beta is 0.985. The sum of alpha and beta in EMA is restricted to equal 1.
#However, the sum of alpha and beta in garch as seen here is not limited to 1.


# In[21]:


model = arch_model(data_2['Daily Returns'], vol='Garch', mean='constant', p=1, q=1, dist = 'normal')
result = model.fit()
plt.figure()
plt.plot(result.conditional_volatility, color='green')
plt.title('Conditional Standard Deviation of GARCH(1,1)')
plt.xlabel('Date')
plt.ylabel('Conditional Standard Deviation')
plt.grid(True)
plt.show()


# In[22]:


gjr_model = arch_model(data_2['Daily Returns'], vol='Garch', p=1, o=1, q=1)
res_gjr = gjr_model.fit()
print(res_gjr)

#Testing the hypothesis that gamma=0 provides a test of asymmetric effect of current and past squared returns on volatility.
#The gamma of this model is 0.1404. Gamma is not equal to 0, and thus the hypothesis is rejected.
#There is no evidence that asymmetry is statistcially significant.


# In[23]:


nforecast= 250
nstart= data_2['Daily Returns'].index[-1]

garch_fit= arch_model(data_2['Daily Returns'], mean='AR', lags=1).fit()
garch_forecasts= garch_fit.forecast(horizon=nforecast, start=nstart)

gjr_fit= arch_model(data_2['Daily Returns'], p=1, o=1, q=1, mean='AR', lags=1).fit()
gjr_forecasts= gjr_fit.forecast(horizon=nforecast, start=nstart)

garch_volatility_forecasts = np.sqrt(garch_forecasts.variance.dropna())
gjr_garch_volatility_forecasts = np.sqrt(gjr_forecasts.variance.dropna())

plt.figure()
plt.plot(garch_volatility_forecasts, color='blue', label='Garch(1,1)')
plt.plot(gjr_garch_volatility_forecasts, color='red', label='GJR-Garch(1,1)')
plt.title('Volatility Forecasts Up to 250 Days Ahead')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




