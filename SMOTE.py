#!/usr/bin/env python
# coding: utf-8

# In[1]:


## from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

# In[21]:


import pandas as pd

df = pd.read_csv('3.data.csv')  # 输入需要处理的数据
# df = (df - df.min(axis=0))/(df.max(axis=0) - df.min(axis=0))
df.head(3)

# In[22]:


# 查看样本是否均衡
sns.countplot(df.Y);
plt.xlabel('Y');
plt.ylabel('Number of occurrences');

# In[29]:


# 处理不平衡数据
from imblearn.over_sampling import SMOTE

# Resample the minority class. You can change the strategy to 'auto' if you are not sure.
sm = SMOTE(sampling_strategy='auto', random_state=7)

# Fit the model to generate the data.
oversampled_X, oversampled_Y = sm.fit_resample(df.drop('Y', axis=1), df['Y'])
oversampled_data_1 = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)
oversampled_data_1.columns = df.columns

# In[30]:


from collections import Counter

print(Counter(oversampled_Y))

# In[31]:


# 查看样本是否均衡
sns.countplot(oversampled_Y);
plt.xlabel('Y');
plt.ylabel('Number of occurrences');

oversampled_data_1.to_csv('oversampled_data.csv')  # 导出数据
