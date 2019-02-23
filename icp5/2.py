from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pandas as pd
data= pd.read_csv('College.csv')
data=data.drop(columns=['name','Private'],axis=1)
train,test=train_test_split(data,test_size=0.4)
kmean=KMeans(n_clusters=4,max_iter=300)
kmean.fit(train)
print(kmean.predict(test))