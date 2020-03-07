import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(n_total, n_positive):
    x = np.random.normal(size=(n_total, 2))
    x[:n_positive, 0] -= 2
    x[n_positive:, 0] += 2
    x[:, 1] *= 2.
    x = np.concatenate([x, np.ones(x.shape[0]).reshape(-1,1)], axis=1)
    y = np.empty(n_total, dtype=np.int64)
    y[:n_positive] = 1
    y[n_positive:] = -1
    return x, y



def calc_class_ratio(train_x, train_y, test_x):
    diff_pp = train_x[train_y == 1][None] - train_x[train_y == 1][:, None]
    norm_pp = np.zeros((diff_pp.shape[0], diff_pp.shape[1]))
    for i in range(diff_pp.shape[0]):
        norm_pp[i] = np.linalg.norm(diff_pp[i], ord=2, axis=1)
    A_pp = np.mean(norm_pp.flatten())
    
    diff_pm = train_x[train_y == 1][None] - train_x[train_y == -1][:, None]
    norm_pm = np.zeros((diff_pm.shape[0], diff_pm.shape[1]))
    for i in range(diff_pm.shape[0]):
        norm_pm[i] = np.linalg.norm(diff_pm[i], ord=2, axis=1)
    A_pm = np.mean(norm_pm.flatten())

    diff_mm = train_x[train_y == -1][None] - train_x[train_y == -1][:, None]
    norm_mm = np.zeros((diff_mm.shape[0], diff_mm.shape[1]))
    for i in range(diff_mm.shape[0]):
        norm_mm[i] = np.linalg.norm(diff_mm[i], ord=2, axis=1)
    A_mm = np.mean(norm_mm.flatten())
    
    diff_p = test_x[None] - train_x[train_y == 1][:, None]
    norm_p = np.zeros((diff_p.shape[0], diff_p.shape[1]))
    for i in range(diff_p.shape[0]):
        norm_p[i] = np.linalg.norm(diff_p[i], ord=2, axis=1)
    b_p = np.mean(norm_p.flatten())
    
    diff_m = test_x[None] - train_x[train_y == -1][:, None]
    norm_m = np.zeros((diff_m.shape[0], diff_m.shape[1]))
    for i in range(diff_m.shape[0]):
        norm_m[i] = np.linalg.norm(diff_m[i], ord=2, axis=1)
    b_m = np.mean(norm_m.flatten())
    
    pi_hat = min([1, max(0, (A_pm - A_mm - b_p + b_m)/(2 * A_pm - A_pp - A_mm))])
    
    return pi_hat
    
def cwls(train_x, train_y, test_x):
    pi_hat = calc_class_ratio(train_x, train_y, test_x)
    diag = np.ones(train_x.shape[0]) * pi_hat + (train_y == -1) * (1 - 2 * pi_hat)
    W = np.diag(diag)
    phi = train_x
    theta_weighted = np.linalg.solve(phi.T.dot(W).dot(phi), phi.T.dot(W).dot(train_y[:, None]))
    theta_unweighted = np.linalg.solve(phi.T.dot(phi), phi.T.dot(train_y[:, None]))
    return theta_weighted, theta_unweighted


def visualize(train_x, train_y, test_x, test_y, theta_weighted, theta_unweighted):
    for x, y, name in [(train_x, train_y, 'train'), (test_x, test_y, 'test')]:
        plt.clf()
        plt.figure(figsize=(6, 6))
        plt.xlim(-5., 5.)
        plt.ylim(-7., 7.)
        lin = np.array([-5., 5.])
        plt.title(name)
        plt.plot(lin, -(theta_weighted[2] + lin * theta_weighted[0]) / theta_weighted[1], label='weighted')
        plt.plot(lin, -(theta_unweighted[2] + lin * theta_unweighted[0]) / theta_unweighted[1], linestyle='dashdot', label='unweighted')
        plt.legend(loc='upper right')
        plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1],
                    marker='$O$', c='blue')
        plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1],
                    marker='$X$', c='red')
        plt.savefig('lecture9-h3-{}.png'.format(name))


train_x, train_y = generate_data(n_total=100, n_positive=90)
eval_x, eval_y = generate_data(n_total=100, n_positive=10)
theta_weighted, theta_unweighted = cwls(train_x, train_y, eval_x)
visualize(train_x, train_y, eval_x, eval_y, theta_weighted, theta_unweighted)
