from __future__ import division
from __future__ import print_function

import math
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class GaussianKernelManager:
    @staticmethod
    # 計画行列を計算
    def calc_design_matrix(uni_func, x, c, h):
        return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))


class ExtendedLagrangianManager:
    @staticmethod
    # パラメータの更新
    def renew_params(sample_size, y, phi, l, z_before, u_before):
        theta = np.linalg.solve(
            phi.T.dot(phi) + np.identity(sample_size),
            phi.T.dot(y[:, None]) + z_before - u_before,
        )
        z = np.max(np.concatenate([np.zeros(sample_size)[:, None], theta + u_before - l * np.ones(sample_size)[:, None]], axis=1), axis=1)\
            + np.min(np.concatenate([(np.zeros(sample_size))[:, None], theta + u_before + l * np.ones(sample_size)[:, None]], axis=1), axis=1)
        z = z[:, None]
        u = u_before + theta - z
        return theta, z, u

    @staticmethod
    # L(θ,z,u)の計算
    def calc_extended_lagrangian(uni_func, y, phi, l, theta, z, u):
        return 1/2 * np.linalg.norm((phi * theta - y), ord=2) + l * np.linalg.norm(z, ord=1) + u.T.dot(theta - z) + 1/2 * np.linalg.norm(theta - z, ord=2)


class SparseRegressionConductor:
    def __init__(self, func, xmin, xmax, sample_size, l, h):
        self.uni_func = np.vectorize(func)
        self.sample_size = sample_size
        self.x, self.y = self.generate_sample(xmin, xmax, sample_size)
        self.l = l
        self.h = h
        self.phi = GaussianKernelManager.calc_design_matrix(self.uni_func, self.x, self.x, self.h)

        # 交互方向乗数法のパラメータθ,z,uの初期化
        self.theta = np.ones(sample_size)[:, None]
        self.z = np.ones(sample_size)[:, None]
        self.u = np.ones(sample_size)[:, None]

    def generate_sample(self, xmin, xmax, sample_size):
        x = np.linspace(start=xmin, stop=xmax, num=sample_size)
        target = self.uni_func(x)
        noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
        return x, target + noise

    # 回帰を実行
    def exec(self):
        while(True):
            # 直前のパラメータを保存
            theta_before = self.theta
            z_before = self.z
            u_before = self.u

            self.theta, self.z, self.u = ExtendedLagrangianManager.renew_params(self.sample_size, self.y, self.phi, self.l, z_before, u_before)

            # 収束判定
            L = ExtendedLagrangianManager.calc_extended_lagrangian(self.uni_func, self.y, self.phi, self.l, self.theta, self.z, self.u)
            L_before = ExtendedLagrangianManager.calc_extended_lagrangian(self.uni_func, self.y, self.phi, self.l, theta_before, z_before, u_before)
            if abs(L - L_before) < 1e-7:
                print("0成分は{}個中{}個".format(self.sample_size, np.sum(np.abs(self.theta) < 1e-6)))

                # create data to visualize the prediction
                X = np.linspace(start=self.x[0], stop=self.x[-1], num=5000)
                K = GaussianKernelManager.calc_design_matrix(self.uni_func, self.x, X, self.h)
                prediction = K.dot(self.theta)

                # visualization
                plt.clf()
                plt.scatter(self.x, self.y, c='green', marker='o')
                plt.plot(X, prediction, color='blue')
                plt.plot(X, self.uni_func(X), color='red')
                plt.show()
                break


if __name__ == '__main__':
    sparse_regression_couductor = SparseRegressionConductor(lambda x: math.sin(math.pi * x) / (math.pi * x) + 0.1 * x, -3, 3, 50, 0.03, 1)
    sparse_regression_couductor.exec()