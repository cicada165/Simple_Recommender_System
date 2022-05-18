#!/usr/bin/env python
# coding: utf-8

# # DSW Assignment 2

# ##### Yue (Billy) Liu (yl992)

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import svd
from scipy.sparse.linalg import svds


# ## Part A

# In[2]:


rating = pd.read_csv("user-business.csv",header = None) #Row: user Column:Business
name = pd.read_csv("business.csv", header = None)


# In[3]:


#Excluding entries of the first 100 businesses
tmp = rating.iloc[:,100:]

#cosine-similarity between users
distance_mx = cosine_similarity(tmp)
#We are only interested in cos-sim(x,alex)
cos_sim = distance_mx[3]
cos_sim = np.delete(cos_sim,3)

#Excluding ratings from alex
R = rating.drop(3, axis=0)

#print(distance_mx.shape)
#print(cos_sim.shape)
#print(R.shape)

#Using the formula provided
score = cos_sim.dot(R)


# In[4]:


ranking = pd.DataFrame({'Score':score[:100], 'Business Name': np.array(name.iloc[:100,:]).flatten()})
ranking = ranking.sort_values(by='Score',ascending=False)
ranking.head(5)


# ##### Top 5 business: 1. Papi's Cuban & Caribbean Grill (43.04) 2. Seven Lamps (33.60) 3. Loca Luna (33.26) 4. Farm Burger (32.78) 5. Piece of Cake (12.62)

# ## Part B

# In[5]:


#Excluding entries of alex
tmp = rating.drop(3,axis=0)
#cosine-similarity between businesses
distance_mx = cosine_similarity(tmp.T)

#We only want alex's rating to business
R = np.array(rating.iloc[[3]]).flatten()

#print(distance_mx.shape)
#print(R.shape)

#Using the formula provided
score = distance_mx.dot(R)


# In[6]:


ranking = pd.DataFrame({'Score':score[:100], 'Business Name': np.array(name.iloc[:100,:]).flatten()})
ranking = ranking.sort_values(by='Score',ascending=False)
ranking.head(5)


# ##### Top 5 business: 1. Papi's Cuban & Caribbean Grill (6.81) 2. Farm Burger (6.56) 3. Seven Lamps (6.44) 4. Loca Luna (5.85) 5. Piece of Cake (3.73)

# ## Part C

# In[7]:


#Perform SVD, k set to 10
u, s, vh = svds(np.array(rating).astype(float),k=10)
#Using the formula provided
R_mx = u.dot(np.diag(s)).dot(vh)
#print(R_mx.shape)
score_alex = R_mx[3]


# In[8]:


ranking = pd.DataFrame({'Score':score_alex[:100], 'Business Name': np.array(name.iloc[:100,:]).flatten()})
ranking = ranking.sort_values(by='Score',ascending=False)
ranking.head(5)


# ##### Top 5 business: 1. Papi's Cuban & Caribbean Grill (1.19) 2. Loca Luna (0.88) 3. Farm Burger (0.86) 4. Seven Lamps (0.82) 5. Piece of Cake (0.30)

# ## Part D

# In[ ]:




