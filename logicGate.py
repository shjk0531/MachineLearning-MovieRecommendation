import numpy as np
from numerical_dervative import numerical_dervative
from externalFunction import *


class LogicGate:
    def __init__(self, gate_name, xdata, tdata):

        self.name = gate_name

        # 입력 데이터
        # 입력 데이터는 (0,0), (0,1), (1,0), (1,1) 총 4가지
        self.__xdata = xdata.reshape(4, 2)
        self.__tdata = tdata.reshape(4, 1)

        # 가중치 W, 바이어스 b 초기화
        self.__W = np.random.rand(2, 1)  # weight, 2X1 matrix
        self.__b = np.random.rand(1)

        # 학습률 learning rate 초기화
        self.__learning_rate = 1e-2

    # 손실 함수
    def __loss_func(self):
        delta = 1e-7    # log 무한대 발산 방지

        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)

        # cross-entropy
        return -np.sum(self.__tdata*np.log(y + delta) + (1-self.__tdata)*np.log((1-y)+delta))

    # 손실 값 계산
    def error_val(self):
        delta = 1e-7    # log 무한대 발산 방지

        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)

        # cross-entropy
        return -np.sum(self.__tdata*np.log(y + delta) + (1-self.__tdata)*np.log((1-y)+delta))

    # 수치미분을 이용하여 손실함수가 최소가 될 때까지 학습하는 함수

    def train(self):
        def f(x): return self.__loss_func()
        print("Initail error value =", self.error_val())

        for step in range(8001):
            self.__W -= self.__learning_rate * numerical_dervative(f, self.__W)
            self.__b -= self.__learning_rate * numerical_dervative(f, self.__b)

            if (step % 400 == 0):
                print("step = %4d\terror value = %f" %
                      (step, self.error_val()))

    # 미래 값 예측 함수

    def predict(self, input_data):
        z = np.dot(input_data, self.__W) + self.__b
        y = sigmoid(z)

        if y > 0.5:
            result = 1  # True
        else:
            result = 0  # False

        return y, result


xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# AND Gate 검증
tdata = np.array([0, 0, 0, 1])

AND_obj = LogicGate("AND_GATE", xdata, tdata)

AND_obj.train()

print(AND_obj.name, "\n")

for input_data in test_data:
    (sigmoid_val, logical_val) = AND_obj.predict(input_data)
    print(input_data, "=", logical_val, "\n")


# OR Gate 검증
tdata = np.array([0, 1, 1, 1])  # OR 정답데이터

OR_obj = LogicGate("OR_Gate", xdata, tdata)
OR_obj.train()

# OR Gate prediction
print(OR_obj.name, "\n")


for input_data in test_data:
    (sigmoid_val, logical_val) = OR_obj.predict(input_data)
    print(input_data, "=", logical_val, "\n")


# NAND Gate 검증
tdata = np.array([1, 1, 1, 0])  # NAND 정답데이터

NAND_obj = LogicGate("NAND_Gate", xdata, tdata)
NAND_obj.train()


# NAND Gate prediction
print("\n", NAND_obj.name, "\n")

for input_data in test_data:
    (sigmoid_val, logical_val) = NAND_obj.predict(input_data)
    print(input_data, "=", logical_val, "\n")


# XOR 을 NAND + OR => AND 조합으로 계산함
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

s1 = []  # NAND 출력
s2 = []  # OR 출력

new_input_data = []  # AND 입력
final_output = []   # AND 출력

for index in range(len(input_data)):
    s1 = NAND_obj.predict(input_data[index])    # NAND 출력
    s2 = OR_obj.predict(input_data[index])      # OR 출력

    new_input_data.append(s1[-1])   # AND 입력
    new_input_data.append(s2[-1])   # AND 입력

    (sigmoid_val, logical_val) = AND_obj.predict(np.array(new_input_data))

    final_output.append(logical_val)    # AND 출력, 즉 XOR 출력
    new_input_data = []     # AND 입력 초기화

print("XOR Gate")

for index in range(len(input_data)):
    print(input_data[index], "=", final_output[index], end='')
    print("\n")
