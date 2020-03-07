import numpy as np
import scipy.linalg 
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# np.random.seed(1)


def data_generation1(n=100):
    return np.concatenate([np.random.randn(n, 1) * 2, np.random.randn(n, 1)],
                          axis=1)


def data_generation2(n=100):
    return np.concatenate([np.random.randn(n, 1) * 2, 2 * np.round(
        np.random.rand(n, 1)) - 1 + np.random.randn(n, 1) / 3.], axis=1)


def lpp(x, n_components=1):
    x = x - np.mean(x, axis=0)

    W = np.zeros((x.shape[0], x.shape[0]))
    diff = x[None] - x[:, None]
    for i in range(W.shape[0]):
        W[i] = np.exp(-np.linalg.norm(diff[i], ord=2, axis=1) ** 2)
        
    D = np.diag(W.sum(axis=1))
    L = D - W
    
    A = x.T.dot(L).dot(x)
    B = x.T.dot(D).dot(x)
    
    w, v = scipy.linalg.eigh(A,B)
    indexs = np.argsort(w)
    w = w[indexs]
    v = v[indexs, :]
    
    return w[:n_components], v[:n_components, :]


n = 200
n_components = 1
x = data_generation1(n)
# x = data_generation2(n)
w, v = lpp(x, n_components)


plt.xlim(-6., 6.)
plt.ylim(-6., 6.)
plt.plot(x[:, 0], x[:, 1], 'rx')
plt.plot(np.array([-v[:, 0], v[:, 0]]) * 10**4, np.array([-v[:, 1], v[:, 1]]) * 10**4)
plt.savefig('report10.png')
