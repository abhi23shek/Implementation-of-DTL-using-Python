#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from collections import Counter
import math


# In[9]:


data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}


# In[10]:


def entropy(class_labels):
    class_counts = Counter(class_labels)
    entropy_val = 0
    total_samples = len(class_labels)
    
    for count in class_counts.values():
        probability = count / total_samples
        entropy_val -= probability * math.log2(probability)
    
    return entropy_val


# In[11]:


def information_gain(data, attribute, target_attribute):
    attribute_values = set(data[attribute])
    target_entropy = entropy(data[target_attribute])
    weighted_entropy = 0
    
    for value in attribute_values:
        subset_indices = [i for i, val in enumerate(data[attribute]) if val == value]
        subset_class_labels = [data[target_attribute][i] for i in subset_indices]
        weighted_entropy += (len(subset_class_labels) / len(data[target_attribute])) * entropy(subset_class_labels)
    
    return target_entropy - weighted_entropy


# In[12]:


target_attribute = 'PlayTennis'
attributes = [attr for attr in data.keys() if attr != target_attribute]


# In[13]:


for attribute in attributes:
    gain = information_gain(data, attribute, target_attribute)
    print(f"Attribute: {attribute}, Gain: {gain:.4f}")


# In[ ]:




