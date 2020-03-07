
import csv
import numpy as np
import pandas as pd
import seaborn as sn
from IPython.display import display
from sklearn.metrics import accuracy_score

# 前処理
train_classified = []
test_classified = []
for i in range(0, 10):
    with open("../digit/digit_train{}.csv".format(i)) as f:
        sequences = csv.reader(f)
        data = []
        for row in sequences:
            data.append(row)
        train_classified.append(np.stack(data))
    with open("../digit/digit_test{}.csv".format(i)) as f:
        sequences = csv.reader(f)
        data = []
        for row in sequences:
            data.append(row)
        test_classified.append(np.stack(data))
train_classified = np.stack(train_classified).astype(np.float32) # shape:(クラス数=10, クラス毎のデータ数=500, データサイズ=256)
test_classified = np.stack(test_classified).astype(np.float32) # shape:(クラス数=10, クラス毎のデータ数=200, データサイズ=256)

# 訓練データを用いる個数を減らす
train_x = []
for i in range(0, 10):
    train_x.append(train_classified[i, np.random.choice(np.arange(train_classified.shape[1]), 100), :])
train_x = np.stack(train_x).reshape(100 * 10, -1) # shape:(データ数=100×10, データサイズ=256)

test_x = test_classified.reshape([-1, test_classified.shape[2]]) # shape:(データ数=200×10, データサイズ=256)
test_t = np.array([np.full(test_classified.shape[1], i) for i in range(0, 10)]).flatten() # shape:(データ数=200×10,)

# ガウスカーネルを用いて計画行列を作成
def build_design_mat(x1, x2, bandwidth):
    return np.exp(
        -np.sum((x1[:, None] - x2[None]) ** 2, axis=-1) / (2 * bandwidth ** 2))


# 最小二乗法によりパラメータθについて解く
def optimize_param(design_mat, y, regularizer):
    return np.linalg.solve(
        design_mat.T.dot(design_mat) + regularizer * np.identity(len(y)),
        design_mat.T.dot(y))


if __name__ == '__main__':
    design_mat = build_design_mat(train_x, train_x, bandwidth=1.)
    
    # 一対他法により10クラス分類問題を解く
    y = []
    for i in range(0, 10):
        train_t_two_classified = np.where(np.arange(train_x.shape[0]) // (train_x.shape[0] // 10) == i, 1, -1)
        theta = optimize_param(design_mat, train_t_two_classified, regularizer=0.01)
        y.append(build_design_mat(train_x, test_x, bandwidth=1.).T.dot(theta))
    y = np.array(y)
    predict = np.argmax(y, axis=0)
    
    print('accuracy:',accuracy_score(test_t, predict))
    
    table = pd.DataFrame({})
    predict = predict.reshape(10, -1)
    columns = []
    for i in range(0, 10):
        column = [np.sum(predict[i] == j) for j in range(0, 10)]
        columns.append(column)
    for i in range(0, len(columns)):
        table['訓練ラベル{}'.format(i)] = columns[i]
    cm = sn.light_palette("blue", as_cmap=True)
    table = table.style.background_gradient(cmap=cm)
    display(table)

    
