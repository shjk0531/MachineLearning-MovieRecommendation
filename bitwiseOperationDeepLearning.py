from externalFunction import *
from logicGate import *


xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
tdata = np.array([0, 1, 1, 0])

and_obj = LogicGate("XOR", xdata, tdata)

and_obj.train()

test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

for data in test_data:
    print(and_obj.predict(data))

# xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# tdata = np.array([0, 1, 1, 0])

# XOR_obj = LogicGate("XOR_GATE", xdata, tdata)
# XOR_obj.train()

# XOR_obj.predict()
