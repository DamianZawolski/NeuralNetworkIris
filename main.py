import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Weights
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        print(self.W1)
        print(self.W2)

    def forward(self, X):
        # Forward propagation
        self.Z1 = np.dot(X, self.W1)
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2)
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


# Initialize the network
network = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)

# Make a prediction for a sample input
X = np.array([[1, 2]])
prediction = network.forward(X)
print(prediction)