from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(0)  # set the random seed for reproducibility


def generate_sample(xmin, xmax, sample_size):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    pix = np.pi * x
    target = np.sin(pix) / pix + 0.1 * x
    noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
    return x, target + noise


def calc_design_matrix(x, c, h):
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))


# 与えられたベクトルの第i成分を除いたベクトルを返す関数
def delete_ingredient(v, i):
    return v[[j for j in range(0, len(v)) if not j == i - 1]]


# i番目を除いた標本から計画行列を計算
def calc_design_matrix_hat(x, i, h):
    x_divided = delete_ingredient(x, i)
    return calc_design_matrix(x, x_divided, h)


# i番目を除いた標本から計算した計画行列を用いてパラメータを導出
def get_params_hat(y, i, phi_hat, l):
    y_divided = delete_ingredient(y, i)
    theta = np.linalg.solve(
        phi_hat.T.dot(phi_hat) + l * np.identity(np.shape(phi_hat)[1]),
        phi_hat.T.dot(y_divided[:, None])
    )
    return theta


# 交差確認法による二乗誤差を計算
def calc_square_error(x, y, l, h):
    sample_size = len(x)
    phi = calc_design_matrix(x, x, h)
    errors = []
    for i in range(1, sample_size + 1):
        phi_hat = calc_design_matrix_hat(x, i, h)
        theta = get_params_hat(y, i, phi_hat, l)
        errors.append(phi[i - 1, :].T.dot(theta) - y[i - 1])
    return (sum([error**2 for error in errors]) / sample_size)[0]


if __name__ == '__main__':
    # サンプルの生成
    sample_size = 50
    xmin, xmax = -3, 3
    x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)

    # 検証するl,hの値のリスト
    list_l = [0.03, 0.3, 3]
    list_h = [0.1, 1, 10]

    errors = np.zeros((3,3))
    for i, l in enumerate(list_l):
        for j, h in enumerate(list_h) :
            errors[i, j] = (calc_square_error(x, y, l, h))

    print(errors)
