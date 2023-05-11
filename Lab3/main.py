import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation



# zadanie  1

#
# year = np.array([2000, 2002, 2005, 2007, 2010])
# percentage = np.array([6.5, 7.0, 7.4, 8.2, 9.0])
#
#
# # Normalizacja danych
# year_norm = (year - np.mean(year)) / np.std(year)
# percentage_norm = (percentage - np.mean(percentage)) / np.std(percentage)
#
#
# # Definicja funkcji gradientu
# def gradient_descent(x, y, alpha, num_iterations):
#     theta0 = 0
#     theta1 = 0
#     m = len(x)
#
#     for i in range(num_iterations):
#         h = theta0 + theta1 * x
#         theta0 = theta0 - alpha * (1 / m) * np.sum(h - y)
#         theta1 = theta1 - alpha * (1 / m) * np.sum((h - y) * x)
#
#     return theta0, theta1
#
#
# # Definicja funkcji przewidujacej kiedy przekroczymy dany procent
# def predict(starting_year, wanted_percentage, slope, intercept):
#
#     result = slope * starting_year + intercept
#
#     while result <= wanted_percentage:
#         starting_year += 1
#         result = slope * starting_year + intercept
#
#     return starting_year
#
#
# def linear_regression(year, percentage):
#     n = np.size(year)
#     mean_year, mean_percentage = np.mean(year), np.mean(percentage)
#     ss_year_percentage = np.sum(np.multiply(percentage, year) - n * mean_percentage * mean_year)
#     ss_year_year = np.sum(np.multiply(year, year) - n * mean_year * mean_year)
#     if ss_year_year == 0:
#         return None
#     else:
#         b_1 = ss_year_percentage / ss_year_year
#         b_0 = mean_percentage - b_1 * mean_year
#         return b_0, b_1
#
# #
# def animate(i):
#     plt.clf()
#     x_anim = year_norm[:i+1]
#     y_anim = percentage_norm[:i+1]
#     regression_result = linear_regression(x_anim, y_anim)
#     if regression_result is not None:
#         plt.scatter(year_norm[:i+1], percentage_norm[:i+1], color='blue')
#         b_0, b_1 = linear_regression(year_norm[:i+1], percentage_norm[:i+1])
#         regression_line = b_0 + b_1 * year_norm
#         plt.plot(year_norm, regression_line, color='red')
#     plt.title('Linear Regression')
#     plt.xlabel('Year')
#     plt.ylabel('Percentage of Employees')
#
#
# # Wywołanie funkcji gradientu
# theta0, theta1 = gradient_descent(year_norm, percentage_norm, 0.1, 1000)
#
# # Obliczenie współczynników regresji dla danych nieznormalizowanych
# slope = theta1 * (np.std(percentage) / np.std(year))
# intercept = np.mean(percentage) - slope * np.mean(year)
#
# prediction_2012 = slope * 2012 + intercept
# prediction_over_12 = predict(2010, 12, slope, intercept)
#
# # Wypisz równanie regresji liniowej
# print(f"\nLinear regression model: Y = {slope:.3f} * X + {intercept:.3f}")
# print(f"Prediction for year 2012:  {prediction_2012:.3f}")
# print(f"Unemployment will exceed 12% in:  {prediction_over_12}")
#
#
# fig = plt.figure()
# animation = FuncAnimation(fig, animate, frames=len(year), interval=500, repeat=False)
# plt.show()





# zadanie 2


# def step_function(x):
#     return 1 if x >= 0 else 0
#
#
# def perceptron_train(x, y, learning_rate=0.1, epochs=100):
#     weights = np.random.rand(x.shape[1])
#     bias = np.random.rand(1)
#
#     for epoch in range(epochs):
#         for i in range(x.shape[0]):
#             y_hat = step_function(np.dot(x[i], weights) + bias)
#             weights += learning_rate * (y[i] - y_hat) * x[i]
#             bias += learning_rate * (y[i] - y_hat)
#     return weights, bias
#
#
# def perceptron_test(x, weights, bias):
#     y_hat = step_function(np.dot(x, weights) + bias)
#     return y_hat
#
#
# x_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y_and = np.array([0, 0, 0, 1])
#
# x_not = np.array([[0], [1]])
# y_not = np.array([1, 0])
#
# weights_and, bias_and = perceptron_train(x_and, y_and)
# weights_not, bias_not = perceptron_train(x_not, y_not)
#
# test_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# for i in range(test_and.shape[0]):
#     result_and = perceptron_test(test_and[i], weights_and, bias_and)
#     print("AND{} = {}".format(test_and[i], result_and))
#
# print("")
#
# test_not = np.array([[0], [1]])
# for i in range(test_not.shape[0]):
#     result_not = perceptron_test(test_not[i], weights_not, bias_not)
#     print("NOT{} = {}".format(test_not[i], result_not))


############################################################################################


# zadanie 3

np.random.seed(0)

# funkcja sigmoidalna
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# pochodna funkcji sigmoidalnej
def sigmoid_derivative(z):
    return z * (1 - z)


# wejscie i oczekiwane wyjscie
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])


epochs = 10000
lr = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1


# Randomowe wagi i bias
hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
output_bias = np.random.uniform(size=(1, outputLayerNeurons))


print("Initial hidden weights: ", end='')
print(*hidden_weights)
print("Initial hidden biases: ", end='')
print(*hidden_bias)
print("Initial output weights: ", end='')
print(*output_weights)
print("Initial output biases: ", end='')
print(*output_bias)


# Training algorithm
for _ in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(inputs, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Backpropagation - propagacja wsteczna
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr


print("Final hidden weights: ", end='')
print(*hidden_weights)
print("Final hidden bias: ", end='')
print(*hidden_bias)
print("Final output weights: ", end='')
print(*output_weights)
print("Final output bias: ", end='')
print(*output_bias)

print("\nOutput from neural network after 10,000 epochs: ", end='')
print(*predicted_output)


print("\n[x1 x2] -> [y]")
for i in range(len(inputs)):
    print(inputs[i], "->", np.round(predicted_output[i]).astype(int))
