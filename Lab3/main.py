import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.animation as animation

print("\nZESTAW 1 ZADANIE 1 i 2 \n")

X = np.array([[2000], [2002], [2005], [2007], [2010]])
y = np.array([6.5, 7.0, 7.4, 8.2, 9.0])

model = LinearRegression()

model.fit(X, y)

rok = [[2023]]
przewidywany_procent = model.predict(rok)

print("Przewidywany procent bezrobotnych w 2021 roku: {:.3f}%".format(przewidywany_procent[0]))

# ZADANIE 3

print("\nZESTAW 1 ZADANIE 3 \n")

X = np.array([[2000], [2002], [2005], [2007], [2010]])
y = np.array([6.5, 7.0, 7.4, 8.2, 9.0])

model = LinearRegression()

fig, ax = plt.subplots()
ax.scatter(X, y)

line, = ax.plot([], [])

def animate(i):

    model.fit(X, y)
    a = model.coef_[0]
    b = model.intercept_

    y_pred = a * X + b

    line.set_data(X, y_pred)

    return line,

ani = animation.FuncAnimation(fig, animate, frames=10, blit=True)
plt.show()




# ZADANIE1a

print("\nZESTAW 2 ZADANIE 1a \n")

class Perceptron:
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.predict(x)
                e = d[i] - y
                x = np.insert(x, 0, 1)
                self.W = self.W + self.lr * e * x

X = np.array([[0,0], [0,1], [1,0], [1,1]])
d = np.array([0,0,0,1])

perceptron = Perceptron(input_size=2)
perceptron.fit(X, d)

print(perceptron.W)

#ZADANIE1b

print("\nZESTAW 2 ZADANIE 1b \n")

X = np.array([[0], [1]])
d = np.array([1,0])

perceptron = Perceptron(input_size=1)
perceptron.fit(X, d)

print(perceptron.W)

#ZADANIE 2

print("\nZESTAW 2 ZADANIE 2 \n")

class Perceptron:
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.predict(x)
                e = d[i] - y
                x = np.insert(x, 0, 1)
                self.W = self.W + self.lr * e * x

X = np.array([[1,0], [1,1], [0,1], [0,0]])
d = np.array([1,0,0,0])

perceptron = Perceptron(input_size=2)
perceptron.fit(X, d)

print(perceptron.W)

# ZADANIE 3

print("\nZESTAW 2 ZADANIE 3 \n")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    nn = NeuralNetwork(X, y)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()

    print(np.round(nn.output))


# ZADANIE 4

print("\nZESTAW 2 ZADANIE 4 \n")

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(1)
synapse_0 = 2 * np.random.random((2, 4)) - 1
synapse_1 = 2 * np.random.random((4, 1)) - 1

for j in range(60000):

    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))

    layer_2_error = y - layer_2

    if (j % 10000) == 0:
        print("Error: " + str(np.mean(np.abs(layer_2_error))))

    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)
    layer_1_error = layer_2_delta.dot(synapse_1.T)
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    synapse_1 += layer_1.T.dot(layer_2_delta)
    synapse_0 += layer_0.T.dot(layer_1_delta)

print("Wynik po nauczeniu:")
print(layer_2)