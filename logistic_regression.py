import numpy as np
from numerical_dervative import numerical_dervative

x_data = np.array([[2, 4], [4, 11], [6, 6], [8, 5], [10, 7],
                  [12, 16], [14, 8], [16, 3], [18, 7]])
t_data = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1]).reshape(9, 1)

W = np.random.rand(2, 1)  # 2X1 행렬
b = np.random.rand(1)
print("W =", W, ", W.shape =", W.shape, ", b =", b, ", b.shape =", b.shape)


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def loss_func(x, t):
    delta = 1e-7  # log 무한대 발산 방지

    z = np.dot(x, W) + b
    y = sigmoid(z)

    # cross-entropy
    return -np.sum(t*np.log(y + delta) + (1-t)*np.log((1-y)+delta))


def error_val(x, t):
    delta = 1e-7  # log 무한대 발산 방지

    z = np.dot(x, W) + b
    y = sigmoid(z)

    # cross-entropy
    return -np.sum(t*np.log(y + delta) + (1-t)*np.log((1-y)+delta))


def predict(x):
    z = np.dot(x, W) + b
    y = sigmoid(z)

    if y > .5:
        result = 1  # True
    else:
        result = 0  # False

    return y, result


learning_rate = 1e-2    # 1e-2, 1e-3은 손실함수 값 발산


def f(x): return loss_func(x_data, t_data)


print("Initial error value =", error_val(
    x_data, t_data), "Initail W =", W, "\nb =", b)

for step in range(80001):
    W -= learning_rate * numerical_dervative(f, W)
    b -= learning_rate * numerical_dervative(f, b)

    if (step % 500 == 0):
        print("step:", step, "\terror value:", error_val(
            x_data, t_data), "\nW =", W, "\tb =", b)

test_data = np.array([3, 17])
print(predict(test_data))
test_data = np.array([5, 8])
print(predict(test_data))
test_data = np.array([7, 21])
print(predict(test_data))
test_data = np.array([12, 0])
print(predict(test_data))
