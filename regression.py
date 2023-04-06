import numpy as np
from numerical_dervative import numerical_dervative

# 학습데이터(Training Data) 준비
loaded_data = np.loadtxt("datas/data-01-test-score.csv",
                         delimiter=',', dtype=np.float32)

x_data = loaded_data[:, 0:-1]
t_data = loaded_data[:, [-1]]

# 임의의 직선 y = W1x1 + W2x2 + W3x3 + b 정의
W = np.random.rand(3, 1)  # 3X1 행렬
b = np.random.rand(1)
print("W=", W, ", W.shape=", W.shape, ", b=", b, ", b.shape=", b.shape)


# 손실함수 E(W,b) 정의
def loss_func(x, t):
    y = np.dot(x, W) + b

    return (np.sum((t - y)**2) / (len(x)))


# 손실함수 값 계산 함수
# 입력변수 x, t: numpy type
def error_val(x, t):
    y = np.dot(x, W) + b

    return (np.sum((t - y)**2) / (len(x)))


learning_rate = 1e-5    # 1e-2, 1e-3은 손실함수 값 발산


def f(x): return loss_func(x_data, t_data)  # f(x) = loss_func(x_data, t_data)


print("initial error value=", error_val(
    x_data, t_data), "\tinitial W=", W, "\nb=", b)

for step in range(100001):
    W -= learning_rate * numerical_dervative(f, W)
    b -= learning_rate * numerical_dervative(f, b)

    if (step % 400 == 0):
        print("step=", step, "\terror value=", error_val(
            x_data, t_data), "\tW=", W, "b=", b)


def predict(x):
    y = np.dot(x, W) + b
    return y


test_data = np.array([100, 98, 81])
print(predict(test_data))
