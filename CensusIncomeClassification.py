
# coding: utf-8

# # Classify as above 50,000 or below 50,000

# In[7]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[2]:


data = pd.read_csv('census_data.csv')


# In[3]:


data['income_bracket'][32560]


# In[4]:


data.tail()


# In[5]:


data['income_bracket'] = data['income_bracket'].apply(lambda x : 0 if (x == ' <=50K') else 1,) 


# In[6]:


from sklearn.model_selection import train_test_split


# In[13]:


X = data.drop('income_bracket',axis=1)
y = data['income_bracket']


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[15]:


age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)


# In[16]:


feat_cols = [gender,occupation,marital_status,relationship,education,workclass,native_country,
            age,education_num,capital_gain,capital_loss,hours_per_week]


# In[17]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=100,shuffle=True,num_epochs=None)


# In[18]:


model = tf.estimator.LinearClassifier(feature_columns=feat_cols)


# In[19]:


model.train(input_fn=input_func,steps=5000)


# In[21]:


pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),num_epochs=1,shuffle=False)


# In[35]:


pred = model.predict(input_fn=pred_input_func)


# In[36]:


pred1 = list(pred)


# In[39]:


pred1[1]['class_ids'][0]


# In[42]:


final_pred = []
for item in pred1:
    final_pred.append(item['class_ids'][0])


# In[45]:


from sklearn.metrics import classification_report, confusion_matrix


# In[49]:


print(confusion_matrix(y_test,final_pred))
print('\n')
print(classification_report(y_test,final_pred))

