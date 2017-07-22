# -*- coding: utf-8 -*-
"""
Created on Sat Apr 08 00:20:18 2017

@author: Pranav Thombre
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.io as sio

cont=sio.loadmat('ex7data2.mat')
print cont
print type(cont)
X=cont['X']
print X
xs = [x[0] for x in X]
ys = [x[1] for x in X]
plt.scatter(xs, ys)
plt.show()

kmeans3=KMeans(n_clusters=3)
kmeans3.fit(X)
pred3=kmeans3.predict(X);
print pred3
centroids3=kmeans3.cluster_centers_
print centroids3
#plt.plot(X[0,0],X[0,1])
for i in range(300):
    if(pred3[i]==0):
        plt.scatter(X[i,0],X[i,1],color='r')
        
        
for i in range(300):
    if(pred3[i]==1):
        plt.scatter(X[i,0],X[i,1],color='g')
        
for i in range(300):
    if(pred3[i]==2):
        plt.scatter(X[i,0],X[i,1],color='b')
        

plt.scatter(centroids3[0][0],centroids3[0][1],marker='*',s=240,color='black')
plt.scatter(centroids3[1][0],centroids3[1][1],marker='*',s=240,color='black')   
plt.scatter(centroids3[2][0],centroids3[2][1],marker='*',s=240,color='black')  
plt.show()






kmeans=KMeans(n_clusters=5)
kmeans.fit(X)
pred=kmeans.predict(X);
print pred
centroids=kmeans.cluster_centers_
print centroids
#plt.plot(X[0,0],X[0,1])
for i in range(300):
    if(pred[i]==0):
        plt.scatter(X[i,0],X[i,1],color='r')
        
        
for i in range(300):
    if(pred[i]==1):
        plt.scatter(X[i,0],X[i,1],color='g')
        
for i in range(300):
    if(pred[i]==2):
        plt.scatter(X[i,0],X[i,1],color='b')
        
for i in range(300):
    if(pred[i]==3):
        plt.scatter(X[i,0],X[i,1],color='yellow')
        
for i in range(300):
    if(pred[i]==4):
        plt.scatter(X[i,0],X[i,1],color='magenta')
        
plt.scatter(centroids[0][0],centroids[0][1],marker='*',s=240,color='black')
plt.scatter(centroids[1][0],centroids[1][1],marker='*',s=240,color='black')   
plt.scatter(centroids[2][0],centroids[2][1],marker='*',s=240,color='black')  
plt.scatter(centroids[3][0],centroids[3][1],marker='*',s=240,color='black')   
plt.scatter(centroids[4][0],centroids[4][1],marker='*',s=240,color='black')  
plt.show()

kmeans7=KMeans(n_clusters=7)
kmeans7.fit(X)
pred7=kmeans7.predict(X);
print pred7
centroids7=kmeans7.cluster_centers_
print centroids7
#plt.plot(X[0,0],X[0,1])
for i in range(300):
    if(pred7[i]==0):
        plt.scatter(X[i,0],X[i,1],color='r')
        
        
for i in range(300):
    if(pred7[i]==1):
        plt.scatter(X[i,0],X[i,1],color='g')
        
for i in range(300):
    if(pred7[i]==2):
        plt.scatter(X[i,0],X[i,1],color='b')
        
for i in range(300):
    if(pred7[i]==3):
        plt.scatter(X[i,0],X[i,1],color='yellow')
        
for i in range(300):
    if(pred7[i]==4):
        plt.scatter(X[i,0],X[i,1],color='magenta')
for i in range(300):
    if(pred7[i]==5):
        plt.scatter(X[i,0],X[i,1],color='cyan')
        
for i in range(300):
    if(pred7[i]==6):
        plt.scatter(X[i,0],X[i,1],color='orange')
        

        
plt.scatter(centroids7[0][0],centroids7[0][1],marker='*',s=240,color='black')
plt.scatter(centroids7[1][0],centroids7[1][1],marker='*',s=240,color='black')   
plt.scatter(centroids7[2][0],centroids7[2][1],marker='*',s=240,color='black')  
plt.scatter(centroids7[3][0],centroids7[3][1],marker='*',s=240,color='black')   
plt.scatter(centroids7[4][0],centroids7[4][1],marker='*',s=240,color='black')  
plt.scatter(centroids7[5][0],centroids7[5][1],marker='*',s=240,color='black')   
plt.scatter(centroids7[6][0],centroids7[6][1],marker='*',s=240,color='black')  
plt.show()
#psrint type(kmeans)
