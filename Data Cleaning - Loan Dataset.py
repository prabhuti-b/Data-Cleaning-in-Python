#!/usr/bin/env python
# coding: utf-8

# this is a dataset about loan variables which is to be prepared for creating a model over it
# the project includes doing the following as a part of the data cleaning process
# discard the non related variables
# sanitize columns by removing special characters
# convert object to numeric data type
# create dummy variables for categorical variables
# handle NaN values

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


file = r'C:/Users/KASH/Desktop/loan_data_train.csv'


# In[3]:


ld_train = pd.read_csv(file)


# In[5]:


ld_train.info()


# In[6]:


ld_train.isnull().sum()


# In[7]:


ld_train['Amount.Requested'].head(10)
#change to numeric datatype from object datatype


# In[8]:


ld_train['Amount.Funded.By.Investors'].head(10)
#change to numeric datatype from object datatype


# In[9]:


ld_train['Interest.Rate'].head(10)
#remove the % sign & change to numeric datatype from object datatype


# In[10]:


ld_train['Loan.Length'].head(10)
#make dummies


# In[11]:


ld_train['Loan.Purpose'].head(10)
#make dummies


# In[12]:


ld_train['Debt.To.Income.Ratio'].head(10)
#remove the % sign & change to numeric datatype from object datatype


# In[13]:


ld_train['FICO.Range'].head(10)
#it is a range, so impute it with the mean


# In[14]:


ld_train['Open.CREDIT.Lines'].head(10)
#change to numeric datatype from object datatype


# In[15]:


ld_train['Open.CREDIT.Lines'].head(10)
#change to numeric datatype from object datatype


# In[16]:


ld_train['Employment.Length'].head(10)
#change to numeric datatype from categorical datatype


# In[17]:


k=ld_train['FICO.Range'].str.split("-",expand=True).astype(float)
ld_train['fico']=0.5*(k[0]+k[1])
del ld_train['FICO.Range']


# In[18]:


ld_train['fico'].head()


# In[19]:


ld_train.drop(['ID','Amount.Funded.By.Investors'],axis=1,inplace=True)


# In[20]:


ld_train.head(5)


# In[21]:


for col in ['Interest.Rate','Debt.To.Income.Ratio']:
    ld_train[col]=ld_train[col].str.replace("%","")


# In[22]:


for col in ['Amount.Requested', 'Interest.Rate','Debt.To.Income.Ratio','Open.CREDIT.Lines','Revolving.CREDIT.Balance']:
    ld_train[col]=pd.to_numeric(ld_train[col],errors='coerce')


# In[23]:


# Processing Employment.Length
ld_train['Employment.Length']=ld_train['Employment.Length'].str.replace('years','')
ld_train['Employment.Length']=ld_train['Employment.Length'].str.replace('year','')
ld_train['Employment.Length']=np.where(ld_train['Employment.Length'].str[0]=='<',0,ld_train['Employment.Length'])
ld_train['Employment.Length']=np.where(ld_train['Employment.Length'].str[:2]=='10',10,ld_train['Employment.Length'])
ld_train['Employment.Length']=pd.to_numeric(ld_train['Employment.Length'],errors='coerce')


# In[24]:


ld_train.dtypes


# In[25]:


ld_train.shape


# In[26]:


ld_train['Loan.Length'].value_counts()


# In[27]:


ld_train = ld_train[ld_train['Loan.Length'] != '.']


# In[28]:


ld_train['Loan.Length'].value_counts()


# In[29]:


ld_train.shape


# In[30]:


loan_length = pd.get_dummies(ld_train["Loan.Length"],prefix="llen")


# In[31]:


loan_length.head(5)


# In[32]:


ld_train=pd.concat([ld_train,loan_length],1)


# In[33]:


ld_train=ld_train.drop(["Loan.Length"],axis=1)


# In[34]:


ld_train.head(5)


# In[35]:


ld_train['Loan.Purpose'].value_counts()


# In[36]:


loan_purp = pd.get_dummies(ld_train["Loan.Purpose"],prefix="loan_purp")


# In[37]:


ld_train=pd.concat([ld_train,loan_purp],1)


# In[38]:


#ld info
ld_train.info()


# In[39]:


ld_train=ld_train.drop(["Loan.Purpose"],axis=1)


# In[40]:


ld_train['Home.Ownership'].value_counts()


# In[41]:


ld_train = ld_train[ld_train['Home.Ownership'] != 'OTHER']


# In[42]:


ld_train.shape


# In[43]:


ld_train = ld_train[ld_train['Home.Ownership'] != 'NONE']


# In[44]:


ld_train.shape


# In[45]:


hom_own = pd.get_dummies(ld_train["Home.Ownership"],prefix="ownership")


# In[46]:


hom_own.head()


# In[47]:


ld_train=pd.concat([ld_train,hom_own],1)


# In[48]:


ld_train=ld_train.drop(["Home.Ownership"],axis=1)


# In[49]:


ld_train.head()


# In[50]:


ld_train['State'].value_counts()


# In[51]:


ld_train = ld_train[ld_train['State'] != '.']


# In[52]:


ld_train.shape


# In[53]:


state_usa = pd.get_dummies(ld_train["State"],prefix="st")


# In[54]:


state_usa.head()


# In[55]:


ld_train=pd.concat([ld_train,hom_own],1)


# In[56]:


ld_train=ld_train.drop(["State"],axis=1)


# In[57]:


ld_train.head()


# In[58]:


ld_train.shape


# In[59]:


ld_train.info()


# In[60]:


ld_train.isnull().sum()


# In[61]:


ld_train.shape


# In[62]:


nan_rows  = ld_train[ld_train.isna().any(axis=1)]


# In[63]:


nan_rows


# In[64]:


ld_train.head(10)


# In[65]:


ld_train.isnull().sum()


# In[66]:


ld_train.shape


# In[67]:


ld_train['Employment.Length'] = ld_train['Employment.Length'].fillna(ld_train['Employment.Length'].mean())


# In[68]:


ld_train.isnull().sum()


# In[69]:


ld_train['Open.CREDIT.Lines'].mean()


# In[70]:


ld_train['Open.CREDIT.Lines'] = ld_train['Open.CREDIT.Lines'].fillna(ld_train['Open.CREDIT.Lines'].mean())


# In[71]:


nan_rows2  = ld_train[ld_train.isna().any(axis=1)]


# In[72]:


nan_rows2


# In[73]:


ld_train = ld_train.dropna()


# In[74]:


ld_train.isnull().sum()


# In[75]:


ld_train.shape

