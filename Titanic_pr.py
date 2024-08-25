
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


from sklearn.model_selection import train_test_split


# In[3]:


from sklearn.compose import ColumnTransformer


# In[4]:


from sklearn.impute import SimpleImputer


# In[5]:


from sklearn.preprocessing import OneHotEncoder


# In[6]:


from sklearn.preprocessing import MinMaxScaler


# In[7]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[8]:


from sklearn.tree import DecisionTreeClassifier


# In[9]:


from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline


# In[10]:


from sklearn.metrics import accuracy_score


# In[34]:


from sklearn.metrics import r2_score


# In[11]:


df = pd.read_csv('tested.csv')


# In[12]:


df


# In[13]:


df = df.drop(columns=['PassengerId','Name','Ticket','Cabin'],axis=1)


# In[14]:


df


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(df.drop('Survived',axis=1),df['Survived'])


# In[16]:


x_train


# In[17]:


y_train


# In[18]:


x_test


# In[19]:


df.isnull().sum()


# In[20]:


x_train


# In[21]:


#Step 1 and 2
#Applying SimpleImputer to age and fare
#Applying one hot encoder to sex and embarked
trans1 = ColumnTransformer(transformers=[
    ('impute age',SimpleImputer(),[2]),
    ('impute fare',SimpleImputer(),[5]),
    ('encode sex_embarked',OneHotEncoder(sparse=False,handle_unknown='ignore'),[1,6]),
],remainder='passthrough')


# In[22]:


#Step 3
#Scaling
trans3 = ColumnTransformer(transformers=[
    ('scaling',MinMaxScaler(),slice(0,10))
])


# In[23]:


#Step 4
#Feature selection
trans4 = SelectKBest(score_func=chi2,k=10)


# In[24]:


#Step 5
#Training the model
trans5 = DecisionTreeClassifier()


# In[25]:


#Creating a pipeline
pipe = Pipeline([
    ('trf1',trans1),
    ('trf2',trans3),
    ('trf3',trans4),
    ('trf4',trans5)
])


# ## Pipeline VS make_pipeline

# #### Pipeline requires naming of steps make pipeline does not.
# #### Same applies to ColumnTransformer Vs make_column_transformer

# In[26]:


#Alternatively using make_pipeline
pipe = make_pipeline(trans1,trans3,trans4,trans5)


# In[27]:


pipe.fit(x_train,y_train)


# ###### Use fit_transform when model training is not there in the pipeline

# ###### Usse fit when model training is there in the pipeline

# ### 

# ### Exploring the pipeline

# In[28]:


#Display pipeline
from sklearn import set_config
set_config(display='diagram')


# In[29]:


#Predict
y_pred = pipe.predict(x_test)


# In[30]:


y_pred


# In[31]:


accuracy_score(y_pred,y_test)


# In[32]:


pipe.named_steps


# ### Removing feature selection step from the pipeline

# In[35]:


pipe1 = Pipeline([
    ('trf1',trans1),
    ('trf2',trans3),
    ('trf3',trans5)
])


# In[36]:


pipe1.fit(x_train,y_train)


# In[37]:


pipe1.named_steps


# In[38]:


y_pred = pipe1.predict(x_test)


# In[39]:


y_pred


# In[46]:


r2_score(y_pred,y_test)


# ## Cross validation using pipeline

# In[45]:


#Cross validation using cross_val_score
from sklearn.model_selection import cross_val_score


# In[42]:


cross_val_score(pipe, x_train, y_train, cv=5, scoring='accuracy').mean()


# In[43]:


import pickle


# In[47]:


pickle.dump(pipe,open('models/pipe.pkl','wb'))


# In[48]:


x_train

