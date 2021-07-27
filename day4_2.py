from sklearn.datasets.samples_generator import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.cluster import SpectralClustering
from mpl_toolkits import mplot3d
X,y_true=make_moons(300,noise=0.05)
data=pd.read_csv("sample.csv")
X=data.as_matrix()
print(X)
kmeans=KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans=kmeans.predict(X)
print(y_kmeans)
fig=plt.figure()
ax=plt.axes(projection='3d')
ax.scatter3D(X[:,0],X[:,1],X[:,2],c=y_kmeans,s=50,cmap='viridis')
plt.show()
