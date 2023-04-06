import numpy as np
from numerical_dervative import numerical_dervative

x_data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).reshape(10, 1)
t_data = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]).reshape(10, 1)

W = np.random.rand(1, 1)
b = np.random.rand(1)
print("W=", W, ", W.shape=", W.shape, ", b=", b, ", b.shape=", b.shape)


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def loss_func(x, t):
    delta = 1e-7

    z = np.dot(x, W) + b
    y = sigmoid(z)

    # cross-entropy
    return -np.sum(t*np.log(y + delta) + (1-t)*np.log((1-y)+delta))


def error_val(x, t):
    delta = 1e-7    # log 무한대 방지

    z = np.dot(x, W) + b
    y = sigmoid(z)

    # cross-entropy
    return -np.sum(t*np.log(y + delta) + (1-t)*np.log((1-y)+delta))


def predit(x):

    z = np.dot(x, W) + b
    y = sigmoid(z)

    if y > .5:
        result = 1  # True
    else:
        result = 0  # False

    return y, result


learning_rate = 1e-2    # 발산하는 경우, 1e-3 ~ 1e-6 등으로 바꾸어서 실행


# f(x) = loss_func(x_data, t_data)
def f(x): return loss_func(x_data, t_data)


print("Initial error value =", error_val(
    x_data, t_data), "\tInitial W =", W, "\nb =", b)

for step in range(10001):
    W -= learning_rate * numerical_dervative(f, W)
    b -= learning_rate * numerical_dervative(f, b)

    if (step % 500 == 0):
        print("step:", step, "\terror value:", error_val(
            x_data, t_data), "\nW =", W, "\tb =", b)


(real_val, logical_val) = predit(3)

print(real_val, logical_val)

(real_val, logical_val) = predit(17)

print(real_val, logical_val)
