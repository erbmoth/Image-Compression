# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

i=Image.open("img053.jpg")

iar=np.asarray(i)

plt.imshow(iar)
plt.show()

X=np.reshape(iar,(iar.shape[0]*iar.shape[1],3))
print X
kmeans=KMeans(n_clusters=9)
kmeans.fit(X)

centroids=kmeans.cluster_centers_
pred=kmeans.predict(X);

print(centroids.shape)
print(pred.shape)
print centroids
print pred
X_recovered=np.zeros(X.shape)
X_recovered=centroids[pred,:]

Xnew=np.reshape(X_recovered,(iar.shape[0],iar.shape[1],3))

#plt.imshow(Xnew)
plt.show()


