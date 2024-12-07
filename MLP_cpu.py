# Implementação do modelo de rede Multilayer Perceptron e do algoritmo de treinamento Backpropagation
# Baseado em Hagan (2014)
# Halena Kulmann Duarte, 2024

import numpy as np
import matplotlib.pyplot as plt
import random
import time

class MLP:
    def __init__(self, layers, learning_rate=0.01, epochs=10000):
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.biases = self.initialize_MLP()

    def initialize_MLP(self):
        weights = []
        biases = []

        for i in range(1, len(self.layers)):
            weights.append(np.random.uniform(low=-0.5, high=0.5, size=(self.layers[i], self.layers[i-1])))
            biases.append(np.random.uniform(size=(self.layers[i], 1)))

        return weights, biases

    def logsigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def logsigmoid_derivative(self, x):
        return x * (1 - x)

    def purelin(self, x):
        return x

    def purelin_derivative(self, x):
        return 1

    def forward(self, inputs):
        a = inputs

        for i in range(len(self.layers)-1):
            n = self.weights[i] @ a + self.biases[i]
            a = self.logsigmoid(n) if i < len(self.layers) - 2 else self.purelin(n)

        return a

    def forward_pass(self, inputs):
        a = [inputs]

        for i in range(len(self.layers)-1):
            n = self.weights[i] @ a[-1] + self.biases[i]
            activation = self.logsigmoid(n) if i < len(self.layers) - 2 else self.purelin(n)
            a.append(activation)

        return a

    def backward_pass(self, inputs, targets):
        for j in range(len(inputs)):
            activations = self.forward_pass(inputs[j])
            
            sensitivities = []

            e = targets[j] - activations[-1]

            s_L = -2 * self.purelin_derivative(activations[-1]) * e
            sensitivities.append(s_L)

            self.weights[-1] = self.weights[-1] - self.learning_rate * s_L @ activations[-2].T
            self.biases[-1] = self.biases[-1] - self.learning_rate * s_L

            for i in range(len(self.weights)-2, 0-1, -1):
                s_i = np.diag(self.logsigmoid_derivative(activations[i+1]).flatten()) @ self.weights[i+1].T @ sensitivities[-1]
                sensitivities.append(s_i)

                self.weights[i] = self.weights[i] - self.learning_rate * s_i @ activations[i].T
                self.biases[i] = self.biases[i] - self.learning_rate * s_i

    def train(self, inputs, targets):
        for epoch in range(self.epochs):
            self.backward_pass(inputs, targets)
            
            if (epoch + 1) % 1000 == 0:
                total_error = 0

                for pattern in range(len(inputs)):
                    prediction = self.predict(inputs[pattern])
                    error = targets[pattern] - prediction
                    total_error += np.sum(error**2)
                
                mse = total_error / len(inputs)
                print(f"Epoch [{epoch+1}/{self.epochs}], MSE: {mse:.4f}")

                #print(f"Epoch [{epoch}/{self.epochs}]")

    def predict(self, inputs):
        return self.forward(inputs)

# dados de entrada e saída para a função seno
x_train = np.linspace(-2 * np.pi, 2 * np.pi, 360)
y_train = np.sin(x_train)

inputs = [np.array([[x]]) for x in x_train]
targets = [np.array([[y]]) for y in y_train]

# configuração do modelo
layers = [1, 10, 10, 1]
mlp = MLP(layers, learning_rate=0.01, epochs=1000)

# treinamento
start_time = time.perf_counter()

mlp.train(inputs, targets)

elapsed_time = time.perf_counter() - start_time

# predição
predictions = np.array([mlp.predict(inp).flatten()[0] for inp in inputs])

print(f"Tempo total de treinamento: {elapsed_time:.2f} segundos")

plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, label='Valor Real (Seno)', color='blue')
plt.plot(x_train, predictions, label='Predição pela MLP', color='red', linestyle='--')
plt.legend()
plt.title('Aproximação da Função Seno com Rede Neural MLP (NumPy)')
plt.xlabel('Entrada (x)')
plt.ylabel('Saída (y)')
plt.grid(True)
plt.show()
