#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pymysql as sql 
from sklearn.preprocessing import StandardScaler , MinMaxScaler , LabelEncoder


# In[3]:


cnt=sql.connect(
    user='root',
    password='123456',
    host='localhost',
    database='xyz_holidays'
)
cur=cnt.cursor()
cur.execute("select * from travel;")
data=cur.fetchall()
cols=[i[0] for i in cur.description]
TR=pd.DataFrame(data=data,columns=cols)


# In[4]:


#TR.columns


# In[5]:


#TR.head(1)


# In[6]:


#int('')


# In[7]:


TR['Age']=['0' if i=='' else i for i in TR['Age']]


# In[8]:


#for i in TR.columns:
 #   if TR[i].dtypes != 'object':
  #      sns.histplot(TR[i])
   #     plt.show()


# In[9]:


#TR["Gender"]=TR["Gender"].replace("Fe Male","Female")


# In[10]:


#for i in TR.columns:
 #   if TR[i].dtypes=="object":
  #      sns.countplot(x=TR[i])
   #     plt.show()


# In[11]:


#for i in TR.columns:
 #   if TR[i].dtypes!="object":
  #      sns.boxplot(TR[i])
   #     plt.show()


# In[12]:


#for i in TR.columns:
 #   if TR[i].dtypes!="object":
  #      sns.kdeplot(TR[i])
   #     plt.show()


# In[13]:


#TR.isnull().sum()


# 1 replace [Mean | Median | Mode]
# 
# 2 B fill | F fill 
# 
# 3 domain specific knowledge
# 
# 4 interpolation
# 
# 

# In[14]:


for i in TR.columns:
    if TR[i].dtypes=='object':
        TR[i].fillna(TR[i].mode()[0],inplace=True)
    else:
        TR[i].fillna(TR[i].mean(),inplace=True)


# In[15]:


#TR.isnull().sum()


# # Outliers 

# IQR METHOD 

# In[16]:


TR1= TR.copy()


# In[17]:


for i in TR.columns:
    if TR[i].dtypes != 'object' and i != 'ProdTaken':
        Q1 = TR[i].quantile(0.25)
        Q3 = TR[i].quantile(0.75)
        IQR = Q3-Q1
        UF = Q3+(1.5*IQR)
        LF = Q1-(1.5*IQR)
        TR=TR[TR[i].between(LF,UF)]


# In[ ]:





# In[18]:


TR.shape[0]/TR1.shape[0]


# In[ ]:





# # Standardization

# In[19]:


TR2 = TR1.copy()


# In[20]:


STD= StandardScaler()


# In[21]:


for i in TR1.columns:
    if TR1[i].dtypes!='object':
        TR1[i+"std"]= STD.fit_transform(TR1[[i]])
        TR1= TR1[TR1[i+'std'].between(-3,3)]
        TR1.drop(i+'std',axis=1,inplace=True)


# In[22]:


TR1.shape[0] / TR2.shape[0]


# In[23]:


LE =  LabelEncoder()


# In[24]:


np.unique(LE.fit_transform(TR1["TypeofContact"]))


# In[25]:


LE.classes_


# In[26]:


Labels = []
Encodings= []
for i in TR1.columns:
    if TR1[i].dtypes=='object':
        TR1[i]= LE.fit_transform(TR1[i])
        Labels.append(LE.classes_)
        Encodings.append(np.unique(LE.fit_transform(TR1[i])))


# In[27]:


TR1


# In[28]:


Labels=list(zip(TR.select_dtypes('object').columns,Labels))


# In[29]:


list(zip(TR.select_dtypes('object').columns,Encodings))


# In[30]:


Labels


# # Model 

# In[31]:


TR1.drop('CustomerID',axis=1,inplace=True)


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


x= TR1.drop('ProdTaken',axis=1)
y= TR1.ProdTaken


# In[34]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25)


# In[35]:


from sklearn.linear_model import LogisticRegression


# In[43]:


LR=LogisticRegression(max_iter=500)


# In[44]:


LR.fit(xtrain,ytrain)


# In[48]:


LR.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])


# In[49]:


LR.predict(xtest)


# In[51]:


LR.score(xtest,ytest)


# In[52]:


LR.score(xtrain,ytrain)


# In[ ]:





# In[ ]:




