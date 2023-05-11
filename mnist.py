from keras.datasets import mnist

(x_train_data, t_train_data), (x_test_data, t_test_data) = mnist.load_data()

print(x_train_data)
print(t_test_data)
