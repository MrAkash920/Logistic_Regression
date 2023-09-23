#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics
df = pd.read_csv('creditcard.csv')
df.head()


# In[3]:


print("Sahpe of the dataset:", df.shape)


# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# In[57]:


print(df.columns)


# In[56]:


print ('Not Fraud % ',round(df['Class'].value_counts()[0]/len(df)*100,2))
print ('Fraud %    ',round(df['Class'].value_counts()[1]/len(df)*100,2))


# In[8]:


features = df.iloc[:, 1:30].columns
target = df.iloc[:1, 30:].columns

data_features = df[features]
data_target = df[target]


# In[15]:


print(features)
print()
print(target)


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size = 0.70, test_size = 0.30, random_state = 1)


# In[33]:


X_train.shape


# In[34]:


X_test.shape


# In[35]:


y_train.shape


# In[38]:


y_train = np.ravel(y_train)


# In[42]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)


# In[44]:


#Predict Regression
lr.predict(X_test[0:10])


# In[47]:


predictions = lr.predict(X_test)
#Find accuracy score
score = lr.score(X_test, y_test)
print(score)


# In[48]:


#Print Confusion Matrix
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)


# In[49]:


#Heat Map of confusion matrix
plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel ('Actual Label')
plt.xlabel ('Predicted Label')
all_sample_title = 'Acurracy Score:{0}'.format(score)
plt.title(all_sample_title, size=15)


# In[ ]:




