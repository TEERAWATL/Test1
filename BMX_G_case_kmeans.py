# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn import metrics


# In[2]:


data_1 = pd.read_csv("D:/TL/DATA_D/data_Downloads/Msc_stady_DATA_SCIENCE_TU/Advand Python for Data Science/dataset BMX_G/BMX_G.csv")
data_1


# In[3]:


data_1.describe()


# In[4]:


data_1.isnull().sum() #use bmxleg, bmxwaist


# In[5]:


print(data_1.shape)
data_1 = data_1.dropna(subset = ["bmxleg", "bmxwaist"])
print(data_1.shape)


# In[6]:


data_1.isnull().sum()


# In[7]:


plt.scatter(data_1["bmxleg"], data_1["bmxwaist"])
plt.xlabel("bmxleg")
plt.ylabel("bmxwaist")
plt.show()


# In[8]:


k = 3
x = np.array(list(zip(data_1["bmxleg"], data_1["bmxwaist"])))
km_model = KMeans(n_clusters=k)
km_model = km_model.fit(x)
labels = km_model.predict(x)
centroids = km_model.cluster_centers_
c= ["g","b", "y"]
colors = [c[i] for i in labels]
plt.scatter(data_1["bmxleg"], data_1["bmxwaist"], c = colors )
plt.scatter(centroids[:,0], centroids[:,1], marker="o", s=100, c="red")
plt.show()


# In[9]:


data_1["cluster"] = labels
data_1.sample(10)


# In[10]:


data_1[data_1.cluster==0].head()


# In[11]:


data_1[data_1.cluster==1].head()


# In[12]:


avg = pd.DataFrame(centroids, columns = ["bmxleg","bmxwaist"])
avg


# In[13]:


#Using coefficients 

avgs = [] #ตัวแปร list ว่าง
min_k = 2 #เวลาทำ clustering จะเริ่มที่ 2 กลุ่มขึ้นไป

for k in range(min_k, 10): #อยากลองแบ่งกี่กลุ่มกำหนดเองได้
    km = KMeans(n_clusters=k).fit(x)
    s = metrics.silhouette_score(x, km.labels_)
    print("silhouetteCoefficients for k=", k,s)
    avgs.append(s)

s_k = avgs.index(max(avgs)) + min_k
print(" Optimal k is", s_k)


# In[14]:


avgs


# In[15]:


min_k = 2
a = [0.1, 0.3, 0.2, 0.5 ]

s_k = a.index(max(a)) + min_k
print(" Optimal k is", s_k)


# In[16]:


print(data_1["bmxleg"].unique())


# In[17]:


colors


# In[ ]:




