
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(sample_size=90, n_class=3):
    x = (np.random.normal(size=(sample_size // n_class, n_class))
         + np.linspace(-3., 3., n_class)).flatten()
    y = np.broadcast_to(np.arange(n_class),
                        (sample_size // n_class, n_class)).flatten()
    return x, y

def lspc(x, y, l, h, n_class):
    phi_x = np.exp(-(x[None] - x[:,None]) ** 2 / (2 * h ** 2))
    pi_y = np.zeros((len(y), n_class))
    for label in range(n_class):
        pi_y[:, label] = (y == label)
    theta = np.linalg.solve(phi_x.T.dot(phi_x) + l * np.eye(len(x)), phi_x.T.dot(pi_y))
    return theta

def visualize(x, y, theta, h):
    X = np.linspace(-5., 5., num=100)
    K = np.exp(-(x - X[:, None]) ** 2 / (2 * h ** 2))

    plt.clf()
    plt.xlim(-5, 5)
    plt.ylim(-.3, 1.8)
    unnormalized_prob = K.dot(theta)
    
    print(unnormalized_prob)
    
    # 解の補正
    prob = np.where(unnormalized_prob > 0, unnormalized_prob, 0) /  np.sum(np.where(unnormalized_prob > 0, unnormalized_prob, 0), axis=1, keepdims=True)
    
    plt.plot(X, prob[:, 0], c='blue')
    plt.plot(X, prob[:, 1], c='red')
    plt.plot(X, prob[:, 2], c='green')

    plt.scatter(x[y == 0], -.1 * np.ones(len(x) // 3), c='blue', marker='o')
    plt.scatter(x[y == 1], -.2 * np.ones(len(x) // 3), c='red', marker='x')
    plt.scatter(x[y == 2], -.1 * np.ones(len(x) // 3), c='green', marker='v')

    plt.savefig('report7.png')

x, y = generate_data(sample_size=90, n_class=3)
theta = lspc(x, y, l=0.5, h=1., n_class=3)
visualize(x, y, theta, h=1.)
