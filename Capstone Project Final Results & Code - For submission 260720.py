#!/usr/bin/env python
# coding: utf-8

# # Importing datasets

# In[1]:


import pandas as pd
covid=pd.read_csv('C:\\Users\\drgsr\\Desktop\\CAPSTONE\\owid-covid-data.csv')


# # data exploration & cleaning

# In[2]:


covid.info()


# In[3]:


covid.columns


# In[4]:


covid.describe()


# In[5]:


#Obtain the country specific average so as to compare different country parameters(has removed non-numeric iso code,date,test,tests_units)
covid=covid.groupby('location').mean()
covid.columns


# In[6]:


#For Outcome of interest of number of cases & death we have used we have used proportional measures which are more meaningful such as Cases per million & Deaths per million and removed other absolute measures of the same,and also removed absolute measures of testing, dates and iso code column
covid.drop(['total_tests_per_thousand','total_cases','new_cases','total_deaths','new_deaths','new_cases_per_million','new_deaths_per_million','total_tests','new_tests','new_tests_per_thousand','population'],axis=1,inplace=True)
covid


# In[7]:


covid.columns


# In[8]:


index=covid.index
print (index)


# # missing values using KNN imputation

# In[9]:


covid.isnull().sum()


# In[10]:


import numpy as np
import pandas as pd
import scipy as sp
from sklearn.impute import KNNImputer


# In[11]:


covid = pd.DataFrame(covid)


# In[12]:


imputer = KNNImputer(n_neighbors=2)


# In[13]:


covid_filled = imputer.fit_transform(covid)


# In[14]:


covid_filled


# In[15]:


covid_filled = pd.DataFrame(covid_filled)


# In[16]:


covid_filled.head()


# In[17]:


covid_filled.columns=['total_cases_per_million', 'total_deaths_per_million',
       'population_density', 'median_age', 'aged_65_older', 'aged_70_older',
       'gdp_per_capita', 'extreme_poverty', 'cvd_death_rate',
       'diabetes_prevalence', 'female_smokers', 'male_smokers',
       'handwashing_facilities', 'hospital_beds_per_100k']


# In[18]:


covid_filled.head()


# In[19]:


covid_filled.shape


# In[20]:


covid_filled.isnull().sum()


# # treatment of outliers

# In[21]:


df = pd.DataFrame(np.random.randn(100, 3))

from scipy import stats

df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]


# In[22]:


import pandas as pd
import numpy as np
import scipy as sp
#covid_filled = pd.DataFrame(covid_filled)
#z_scores=scipy.stats.zscore(covid_filled)
z_scores = stats.zscore(covid_filled)
#calculate z-scores of `covid_filled`

abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
new_covid_filled = covid_filled[filtered_entries]

print(new_covid_filled)


# # visualization

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


sns.pairplot(new_covid_filled)
new_covid_filled


# # correlations between attributes

# In[26]:


new_covid_filled.corr()


# In[27]:


# pearson's correlation feature selection for numeric input and numeric output
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
# generate dataset
X, y = make_regression(n_samples=100, n_features=100, n_informative=10)
# define feature selection
fs = SelectKBest(score_func=f_regression, k=10)
# apply feature selection
X_selected = fs.fit_transform(X, y)
print(X_selected.shape)


# In[28]:


# Making use of Pearson Correlation for Feature selection
plt.figure(figsize=(12,10))
cor = new_covid_filled.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
plt.savefig('heat_map.png')


# # identification of key attributes 

# In[29]:


# Total deaths per million Correlation 
cor_target = abs(cor["total_deaths_per_million"])
# Feature selection
relevant_features = cor_target[cor_target>0.5]
relevant_features


# # building model

# In[30]:


y=new_covid_filled['total_deaths_per_million']


# In[31]:


y


# In[32]:


X= new_covid_filled [['total_cases_per_million','aged_70_older']]


# In[33]:


X


# In[34]:


#test-train split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[35]:


X_train


# In[36]:


#linear regression
from sklearn.linear_model import LinearRegression


# In[37]:


regression=LinearRegression(fit_intercept=True)


# In[38]:


regression.fit(X_train,y_train)


# In[39]:


y_predict=regression.predict(X_test)


# In[40]:


y_predict


# In[41]:


y_test


# In[42]:


#comparng test & predict
plt.scatter(y_test,y_predict)
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.title("Multiple Linear Regression Modelling")


# In[43]:


k=2
n=len(X_test)


# In[44]:


n


# In[45]:


#evaluation & performance measures
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from math import sqrt
RMSE=float(format(np.sqrt(mean_squared_error(y_test,y_predict)),'.2f'))
MAE=mean_absolute_error(y_test,y_predict)
MSE=mean_squared_error(y_test,y_predict)
r2=r2_score(y_test,y_predict)
adj_r2=1-(1-r2)*(n-1)/(n-k-1)
MAPE=np.mean(np.abs(y_test-y_predict)/y_test)*100


# In[46]:


print('RMSE=',RMSE,'\nMSE=',MSE,'\nMAE=',MAE,'\nr2=',r2,'\nadj_r2=',adj_r2,'\nMAPE=',MAPE)


# In[47]:


#Residuals
sns.distplot((y_test-y_predict),bins=50)


# In[48]:


#Model 2 for comparison (polynomial regression)
y=new_covid_filled['total_deaths_per_million']
X= new_covid_filled [['total_cases_per_million','aged_70_older']]

#Polynomial Regression
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Polynomial Regression model on the Training set
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_poly, y_train)


# In[49]:


# Predicting the Test set results
y_pred = regressor.predict(poly_reg.transform(X_test))
plt.scatter(y_test,y_pred)
plt.ylabel("Pred")
plt.xlabel("Act")
plt.title("Polynomial Linear Regression Modelling")


# In[50]:


k=2
n=len(X_test)


# In[51]:


n


# In[52]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from math import sqrt
RMSE=float(format(np.sqrt(mean_squared_error(y_test,y_pred)),'.2f'))
MAE2=mean_absolute_error(y_test,y_pred)
MSE2=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
adj_r2=1-(1-r2)*(n-1)/(n-k-1)
MAPE=np.mean(np.abs(y_test-y_pred)/y_test)*100


# In[53]:


print('RMSE=',RMSE,'\nMSE=',MSE,'\nMAE=',MAE,'\nr2=',r2,'\nadj_r2=',adj_r2,'\nMAPE=',MAPE)


# In[ ]:




