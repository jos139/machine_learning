
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(sample_size):
    x = np.random.normal(size=(sample_size, 3))
    x[:, 2] = 1.
    x[:sample_size // 2, 0] -= 5.
    x[sample_size // 2:, 0] += 5.
    y = np.concatenate([np.ones(sample_size // 2, dtype=np.int64),
                        -np.ones(sample_size // 2, dtype=np.int64)])
    x[:3, 1] -= 5.
    y[:3] = -1
    x[-3:, 1] += 5.
    y[-3:] = 1
    return x, y


def svm(x, y, l, lr):
    w = np.zeros(3)[:, None]
    prev_w = w.copy()
    for i in range(10 ** 4):
        a = 1 - x.dot(w) * y[:, None] > 0
        subgrad = -y[:, None] * x
        grad = (a * subgrad).sum(axis=0)[:, None] + l * w
        w -= lr * grad
        if np.linalg.norm(w - prev_w) < 1e-3:
            break
        prev_w = w.copy()

    return w


def visualize(x, y, w):
    plt.clf()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.scatter(x[y == 1, 0], x[y == 1, 1])
    plt.scatter(x[y == -1, 0], x[y == -1, 1])
    plt.plot([-10, 10], -(w[2] + np.array([-10, 10]) * w[0]) / w[1])
    plt.show()


x, y = generate_data(200)
w = svm(x, y, l=.1, lr=1.)
visualize(x, y, w)
