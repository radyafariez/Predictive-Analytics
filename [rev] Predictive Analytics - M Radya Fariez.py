#!/usr/bin/env python
# coding: utf-8

# # [Dicoding Submission] Predictive Analytics - M Radya Fariez

# # 1) Import Library

# In[36]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# # 2) Import Dataset from Directory

# In[37]:


df = pd.read_csv('Used_Car_Datasets.csv')


# # 3) Collect Data Info

# In[38]:


df.info()
# Info Statistik
df.describe()


# Collect Possible NaN Info
# * Output menunjukkan tidak ada data yang kosong (NaN)

# In[39]:


df.isnull().sum()


# * Menghilangkan kolom fitur yang tidak berkaitan dengan target
# * Showing data head after drop column

# In[40]:


df = df.drop(['name'], axis = 'columns')

# Setelah drop column
df.head(6)


# # 4) Univariate Analysis

# In[41]:


numerical_features = ['year', 'selling_price', 'km_driven']
categorical_features = ['fuel', 'seller_type', 'transmission','owner']


# * Analisa kategori pada fitur

# In[42]:


df.groupby('fuel')['fuel'].agg('count')


# In[43]:


df.groupby('seller_type')['seller_type'].agg('count')


# In[44]:


df.groupby('transmission')['transmission'].agg('count')


# In[45]:


df.groupby('owner')['owner'].agg('count')


# * Numerical feature and label statistics

# In[46]:


df.hist(bins=50, figsize=(20,15))
plt.show()


# # 5) Multivariate Analysis

# * Mengamati hubungan antar fitur numerik dengan fungsi pairplot()

# In[47]:


sns.pairplot(df, diag_kind = 'kde')


# * Evaluasi skor korelasi fitur numerik dengan fitur target

# In[48]:


plt.figure(figsize=(10, 8))
correlation_matrix = df.corr().round(2)


# * Untuk print nilai di dalam kotak, gunakan parameter annot=True

# In[49]:


sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Korelasi Matriks untuk Fitur Numerik ", size=20)


# In[50]:


# Melihat kolerasi antara fitur kategorik dengan fitur target (harga)
cat_features = df.select_dtypes(include='object').columns.to_list()
 
for col in cat_features:
  sns.catplot(x=col, y="selling_price", kind="bar", dodge=False, height = 4, aspect = 3,  data=df, palette="Set3")
  plt.title("Rata-rata Harga Jual Relatif terhadap - {}".format(col))


# * Categorical -> One Hot Encoding

# In[51]:


from sklearn.preprocessing import  OneHotEncoder

df = pd.concat([df, pd.get_dummies(df['fuel'], prefix='fuel')],axis=1)
df = pd.concat([df, pd.get_dummies(df['seller_type'], prefix='seller_type')],axis=1)
df = pd.concat([df, pd.get_dummies(df['transmission'], prefix='transmission')],axis=1)
df = pd.concat([df, pd.get_dummies(df['owner'], prefix='owner')],axis=1)

df.drop(['fuel','seller_type','transmission','owner'], axis=1, inplace=True)


# In[52]:


df


# # 6) Data Preparation

# In[53]:


sns.pairplot(df[['year','selling_price','km_driven']], plot_kws={"s": 3});


# In[54]:


from sklearn.decomposition import PCA
 
pca = PCA(n_components=3, random_state=123)
pca.fit(df[['year','selling_price','km_driven']])
princ_comp = pca.transform(df[['year','selling_price','km_driven']])

# Proporsi informasi tiap komponen
pca.explained_variance_ratio_.round(3)


# In[55]:


from sklearn.model_selection import train_test_split
 
X = df.drop(["selling_price"],axis =1)
y = df["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=123)


# In[56]:


print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')


# * Normalisasi menggunakan StandardScaler

# In[57]:


from sklearn.preprocessing import StandardScaler
 
numerical_features = ['year', 'km_driven']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features]


# * Mengecek mean dan standar deviasi

# In[58]:


X_train[numerical_features].describe().round(4)


# # 7) Model Development

# * Menyiapkan dataframe untuk analisis model

# In[59]:


models = pd.DataFrame(index=['train_mse', 'test_mse'], 
                      columns=['KNN', 'RandomForest', 'Boosting'])


# * KNN Model Deployment

# In[60]:


from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
 
knn = KNeighborsRegressor(n_neighbors=12)
knn.fit(X_train, y_train)
 
models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)


# * Random Forest Model Deployment

# In[61]:


from sklearn.ensemble import RandomForestRegressor
 
# Membuat model prediksi
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)
 
models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)    


# * Boosting Algorithm Model

# In[62]:


from sklearn.ensemble import AdaBoostRegressor
 
boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)                             
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)


# * Scalling process

# In[63]:


X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])


# In[64]:


mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])
 
# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}


# * Hitung Mean Squared Error masing-masing algoritma pada data train dan test

# In[65]:


for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3 
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3


# # 8) Evaluation

# * Showing Mean Squared Error

# In[66]:


mse


# * Menunjukkan nilai error

# In[67]:


fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)


# * Menunjukkan nilai prediksi hasil dari 3 model yang paling mendekati dengan y_test

# In[68]:


prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)
 
pd.DataFrame(pred_dict)

