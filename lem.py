
import numpy as np
import matplotlib
from scipy.linalg import eigh

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1)


def data_generation(n=1000):
    a = 3. * np.pi * np.random.rand(n)
    x = np.stack(
        [a * np.cos(a), 30. * np.random.random(n), a * np.sin(a)], axis=1)
    return a, x


def LapEig(x, d=2, k=10):
    W = []
    for x_i in x:
        threshold = np.array([np.dot(x_i - x_j, x_i - x_j) for x_j in x]).argsort()[:k]
        knn_list = list(map(lambda x: x in threshold, np.arange(x.shape[0])))
        W.append(knn_list)
    W = np.array(W)
    W = np.where(np.logical_or(W, W.T), 1, 0)
    
    D = np.diag(W.sum(axis=1))
    L = D - W
    
    eig_val, eig_vec = eigh(L, D)
    eig_vec = eig_vec.T
    indexs = np.argsort(eig_val)
    eig_vec = eig_vec[indexs]
    phi = eig_vec[1:d+1]
    
    return phi.T

def visualize(x, z, a):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], c=a, marker='o')
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(z[:, 0], z[:, 1], c=a, marker='o')
    fig.savefig('report.png')


a, x = data_generation()
z = LapEig(x)
visualize(x, z, a)
