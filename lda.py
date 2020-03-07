
import numpy as np
import matplotlib
import scipy.linalg 

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(46)


def generate_data(sample_size=100, pattern='two_cluster'):
    if pattern not in ['two_cluster', 'three_cluster']:
        raise ValueError('Dataset pattern must be one of '
                         '[two_cluster, three_cluster].')
    x = np.random.normal(size=(sample_size, 2))
    if pattern == 'two_cluster':
        x[:sample_size // 2, 0] -= 4
        x[sample_size // 2:, 0] += 4
    else:
        x[:sample_size // 4, 0] -= 4
        x[sample_size // 4:sample_size // 2, 0] += 4
    y = np.ones(sample_size, dtype=np.int64)
    y[sample_size // 2:] = 2
    return x, y


def fda(x, y):
    indexs1, indexs2 = [], []
    for i in range(y.size):
        if y[i] == 1:
            indexs1.append(i)
        else:
            indexs2.append(i)
    n_1 = len(indexs1)
    n_2 = len(indexs2)
    mu_1 = x[indexs1].mean(axis=0)
    mu_2 = x[indexs2].mean(axis=0)
    S_w = np.zeros((2,2))
    S_b = np.zeros((2,2))
    for indexs, mu, n in zip([indexs1, indexs2], [mu_1, mu_2], [n_1, n_2]):
        for i in indexs:
            S_w += ((x[i] - mu)[None]).T.dot((x[i] - mu)[None])
        S_b += n * (mu[None]).T.dot(mu[None])
    w, v = scipy.linalg.eigh(S_b, S_w)
    indexs = np.argsort(w)[::-1]
    w = w[indexs]
    v = v[indexs, :]
    
    return v[0][None]
            
def visualize(x, y, T):
    plt.figure(1, (6, 6))
    plt.clf()
    plt.xlim(-7., 7.)
    plt.ylim(-7., 7.)
    plt.plot(x[y == 1, 0], x[y == 1, 1], 'bo', label='class-1')
    plt.plot(x[y == 2, 0], x[y == 2, 1], 'rx', label='class-2')
    plt.plot(np.array([-T[:, 0], T[:, 0]]) * 10**2,
             np.array([-T[:, 1], T[:, 1]]) * 10**2, 'k-')
    plt.legend()
    plt.savefig('report11_2.png')


sample_size = 100
# x, y = generate_data(sample_size=sample_size, pattern='two_cluster')
x, y = generate_data(sample_size=sample_size, pattern='three_cluster')
T = fda(x, y)
visualize(x, y, T)
