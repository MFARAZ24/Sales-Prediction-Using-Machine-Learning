#!/usr/bin/env python
# coding: utf-8

# # Data Science Internship 
# # Task 02: SALES PREDICTION USING PYTHON
# # M.Faraz Shoaib
# ## Importing Libraries and Data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


sales_data = pd.read_csv('advertising.csv')
sales_data


# ## Data Validation 
# 

# In[3]:


sales_data.describe()


# In[4]:


sales_data.isnull()


# In[5]:


sales_data.info()


# ## Checking The most Correlated Advertisment Type 

# In[6]:


sales_data.corr()


# In[7]:


#%matplotlib inline
fig,axes = plt.subplots(1,3, figsize=(17,5))
axes[0].scatter(sales_data.TV,sales_data.Sales)
axes[0].set_xlabel("TV")
axes[0].set_ylabel("Sales")
axes[0].set_title("TV & Sales Relation")

axes[1].scatter(sales_data.Radio,sales_data.Sales)
axes[1].set_xlabel("Radio")
axes[1].set_ylabel("Sales")
axes[1].set_title("Radio & Sales Relation")

axes[2].scatter(sales_data.Newspaper,sales_data.Sales)
axes[2].set_xlabel("Newspaper")
axes[2].set_ylabel("Sales")
axes[2].set_title("Newspaper & Sales Relation")


# # Since TV is the most Correlated with sales so we will Use TV and Sales Data for Training Our Model for Sales Prediction

# ## Splitting Data and Reshaping it For LinearResgression Model 

# In[8]:


X = sales_data.TV
y = sales_data.Sales
X_reshape = X.values.reshape(-1,1)


# ## Training Model with the Splitted Dataset

# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_reshape,y,test_size = 0.2, random_state = 0)


# In[10]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_reshape,y)
print("Trained")


# ## Predicting the Model  

# In[11]:


predicttion = model.predict(X_test)
print(predicttion)
line = model.coef_*X_reshape + model.intercept_
plt.scatter(sales_data.TV, sales_data.Sales)
plt.xlabel("TV")
plt.ylabel("sales")
plt.plot(X_reshape, line , color ="red")


# ## Predicting for any random value

# In[12]:


print(model.predict([[300]]))


# ## Finding the Possible Error

# In[13]:


from sklearn import metrics
print ("Mean Squared Error" ,metrics.mean_squared_error(y_test,predicttion))
print("Root Mean Squared Error", np.sqrt(metrics.mean_squared_error(y_test,predicttion)))


# # Thank You
