import numpy as np
import pandas as pd
import os
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from numpy import savetxt

# follows the method by by Legendre17:
#https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix


folder = 'Desktop/coordinates/'
files = os.listdir(folder)
for file in files:
    
    if 'txt' in file:
        id_ = file.split('.')[0]
        print('mds solution')
        distances = np.asarray(pd.read_csv(folder+file, header=None).values)
        print(file, distances.shape)
        embedding = MDS(n_components=2,max_iter=1000,verbose=1, eps=1e-4, dissimilarity='euclidean')
        xy = embedding.fit_transform(distances)
        plt.figure(figsize=(10,10))
        plt.scatter(xy[:,0], xy[:,1])
        plt.savefig(folder + 'plots/' +id_ + '_MDS.png')
        plt.show()
        savetxt(folder +  'xy/' + id_ + '_MDS_xy.csv', xy, delimiter=',')
        
        
        print('eig solution')
        M = np.zeros((18,18))
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                M[i,j] = 0.5*(distances[0, j]**2 + distances[i, 0]**2 - distances[i,j]**2)

        u, v = np.linalg.eig(M)

        idx = u.argsort()[::-1]   
        u = u[idx]
        v = v[:,idx]
        u = u[:2] # take 2dims
        v = v[:, :2]
        xy = []
        for i in range(2):
            vecval = v.T[i] * np.sqrt(u[i])
            xy.append(vecval)
        plt.figure(figsize=(10,10))
        xy = np.asarray(xy)
        plt.scatter(xy[0], xy[1])
        plt.savefig(folder + 'plots/' + id_ + '_eig.png')
        plt.show()
        
        savetxt(folder + 'xy/' + id_ + '_eig_xy.csv', xy, delimiter=',')