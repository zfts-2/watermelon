import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn import linear_model


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def J_cost(X, y, beta):
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    beta = beta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return (- y * np.dot(X_hat, beta) + np.log(1 + np.exp(np.dot(X_hat, beta)))).sum()


def gradDesc(X, y, beta, learning_rate, num_iterations, print_cost):
    for i in range(num_iterations):
        grad = gradient(X, y, beta)
        beta = beta - learning_rate * grad
        if (i % 10 == 9) & print_cost:
            print('{}th iteration, cost is {}'.format(i, J_cost(X, y, beta)))
    return beta


def gradient(X, y, beta):
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    beta = beta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    p1 = sigmoid(np.dot(X_hat, beta))
    grad = (-X_hat * (y - p1)).sum(0)
    return grad.reshape(-1, 1)


def logistic_model(X, y, print_cost=False, learning_rate=2, num_iterations=1):
    row, column = X.shape
    beta = np.random.randn(column + 1, 1) * 0.5 + 1
    return gradDesc(X, y, beta, learning_rate, num_iterations, print_cost)


if __name__ == '__main__':
    # dataset process
    matplotlib.rcParams['font.family'] = 'SimHei'
    data_path = "../data/watermelon3_0_Ch.csv"
    data = pd.read_csv(data_path).values
    data[:, 9] = (data[:, 9] == '是')
    X = data[:, 7:9].astype(float)
    y = data[:, 9].astype(int)
    # plot setting
    plt.xlabel('密度')
    plt.ylabel('含糖量')
    plt.scatter(data[:, 7][y == 1], data[:, 8][y == 1], c='k', marker='o')
    plt.scatter(data[:, 7][y == 0], data[:, 8][y == 0], c='r', marker='x')
    # my regression
    beta = logistic_model(X, y, print_cost=True, learning_rate=0.3, num_iterations=1000)
    w1, w2, intercept = beta
    x1 = np.linspace(0, 1)
    y1 = -(w1 * x1 + intercept) / w2
    ax1, = plt.plot(x1, y1, label=r'my')
    # sklearn regression
    lr = linear_model.LogisticRegression(C=1000)
    lr.fit(X, y)
    lr_beta = np.c_[lr.coef_, lr.intercept_]
    print(J_cost(X, y, lr_beta))
    w1_sk, w2_sk = lr.coef_[0, :]
    x2 = np.linspace(0, 1)
    y2 = -(w1_sk * x2 + lr.intercept_) / w2
    ax2, = plt.plot(x2, y2, label=r'sklearn_logistic')
    # show plot
    plt.legend(loc='upper right')
    plt.show()
