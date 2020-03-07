
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)  # set the random seed for reproducibility


class arc:
    def __init__(self, gamma=0.1):
        self.dim = 3
        self.mu = np.zeros((self.dim, 1))
        self.sigma = np.eye(self.dim) 
        self.gamma = gamma
        
    def __call__(self):
        self.train()
        self.visualize()
        
    def fit(self, n=50):
        x = np.random.randn(n, 3)
        x[:n // 2, 0] -= 15
        x[n // 2:, 0] -= 5
        x[1:3, 0] += 10
        x[:, 2] = 1
        y = np.concatenate((np.ones(n // 2), -np.ones(n // 2)))
        index = np.random.permutation(np.arange(n))
        self.X = x[index]
        self.Y = y[index]
        
    def train(self):
        for i in range(self.X.shape[0]):
            x = self.X[i][:, None]
            y = self.Y[i]
            self.mu = self.mu + (y * max(0, 1 - self.mu.T.dot(x) * y) / (x.T.dot(self.sigma).dot(x) + self.gamma)) * self.sigma.dot(x)
            self.sigma = self.sigma - self.sigma.dot(x).dot(x.T).dot(self.sigma) / (x.T.dot(self.sigma).dot(x) + self.gamma)
        
    def visualize(self):
        mu = self.mu.flatten()
        X = self.X
        Y = self.Y
        plt.clf()
        plt.scatter(X[Y == 1, 0], X[Y == 1, 1])
        plt.scatter(X[Y == -1, 0], X[Y == -1, 1])
        plt.ylim([-2,2])
        plt.plot([-20, 0], -(mu[2] + np.array([-20, 0]) * mu[0]) / mu[1], color='green')
        plt.show()
        

if __name__ == '__main__':
    arc = arc()
    arc.fit()
    arc()
