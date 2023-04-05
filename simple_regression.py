import numpy as np
from numerical_dervative import numerical_dervative

x_data = np.array([1, 2, 3, 4, 5]).reshape(5, 1)
t_data = np.array([2, 3, 4, 5, 6]).reshape(5, 1)

W = np.random.rand(1, 1)
b = np.random.rand(1)
print("W=", W, ", W.shape= ", W.shape, " , b= ", b, ", b.shape= ", b.shape)


def loss_func(x, t):
    y = np.dot(x, W) + b

    return (np.sum((t - y)**2)) / (len(x))


# 손실함수 값 계산 함수
# 입력변수 x, t: numpy type


def error_val(x, t):
    y = np.dot(x, W) + b

    return (np.sum((t - y)**2)) / (len(x))

# 학습을 마친 후, 임의의 데이터에 대해 미래 값 예측 함수
# 입력변수 x: numpy type


def predict(x):
    y = np.dot(x, W) + b

    return y


learning_rate = 1e-2    # 발산하는 경우, 1e-3 ~ 1e-6 등으로 바꾸어서 실행


# f(x) = loss_func(x_data, t_data)
def f(x): return loss_func(x_data, t_data)


print("initial error value= ", error_val(
    x_data, t_data), "initial W=", W, "\n", ", b=", b)

for step in range(8001):
    W -= learning_rate * numerical_dervative(f, W)
    b -= learning_rate * numerical_dervative(f, b)

    if (step % 400 == 0):
        print("step=", step, "\terror value=", error_val(
            x_data, t_data), "\tW=", W, ", b=", b)

print(predict(43))
