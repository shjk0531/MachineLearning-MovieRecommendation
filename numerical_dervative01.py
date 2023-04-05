import numpy as np


def numerical_dervative(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x)

    print("debug 1. initial input variable =", x)
    print("debug 2. initial grad =", grad)
    print("================================================")

    it = np.nditer(x, flags=['multi_index'], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index

        print("debug 3. idx =", idx, ", x[idx] =", x[idx])

        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)  # f(x+delta_x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x)  # f(x-delta_x)
        grad[idx] = (fx1 - fx2) / (2*delta_x)

        print("debug 4. grad[idx] =", grad[idx])
        print("debug 5. grad =", grad)
        print("================================================")

        x[idx] = tmp_val
        it.iternext()

    return grad


# f(x) = x^2, f'(3.0)
def func1(input_obj):
    x = input_obj[0]

    return x ** 2


# # x = 3.0 에서의 편미분 값
# numerical_dervative(func1, np.array([3.0]))


# f(x,y) = 2x + 3xy + y^3
def func2(input_obj):
    x = input_obj[0]
    y = input_obj[1]

    return (2*x + 3*x*y + np.power(y, 3))


# numerical_dervative(func2, np.array([1.0, 2.0]))


# f(w,x,y,z) = wx + xyz + 3w + zy^2, f'(1.0, 2.0, 3.0, 4.0)
def func3(input_obj):
    w = input_obj[0, 0]
    x = input_obj[0, 1]
    y = input_obj[1, 0]
    z = input_obj[1, 1]

    return (w*x + x*y*z + 3*w + z*np.power(y, 2))


numerical_dervative(func3, np.array([[1.0, 2.0], [3.0, 4.0]]))
