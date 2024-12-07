# Implementação do modelo de rede Multilayer Perceptron e do algoritmo de treinamento Backpropagation através da biblioteca PyTorch
# Halena Kulmann Duarte, 2024

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

# verificando se CUDA (GPU) está disponível
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Usando dispositivo: {device}')

# define o modelo MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # estrutura da rede
        self.fc1 = nn.Linear(1, 10)  # camada de entrada (1, 10)
        self.fc2 = nn.Linear(10, 10)  # camada oculta (10)
        self.fc3 = nn.Linear(10, 1)   # camada de saída (1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # função de ativação Sigmoid na primeira camada
        x = torch.sigmoid(self.fc2(x))  # função de ativação Sigmoid na segunda camada
        x = self.fc3(x)  # camada de saída (função de ativação Linear)

        return x

# criando dados de entrada e saída para a função seno
x_train = np.linspace(-2 * np.pi, 2 * np.pi, 360)
y_train = np.sin(x_train)

# convertendo para tensores do PyTorch
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# movendo os tensores para o dispositivo (no caso queremos o CUDA (GPU))
x_train_tensor = x_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)

# inicializando o modelo, a função de perda (erro) e o otimizador
model = MLP().to(device)  # movendo o modelo para o dispositivo
criterion = nn.MSELoss()  # função de perda (erro quadrático médio)
optimizer = optim.Adam(model.parameters(), lr=0.01)  # otimizador com taxa de aprendizado de 0.01

start_time = time.perf_counter()

# treinando o modelo
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()  # modo de treinamento
    
    # forward pass
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)  # Calculando a perda
    
    # backward pass e otimização (pesos)
    optimizer.zero_grad()  # zera os gradientes anteriores
    loss.backward()  # calcula o gradiente
    optimizer.step()  # atualiza os pesos

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], MSE: {loss.item():.4f}')
        #print(f'Epoch [{epoch+1}/{num_epochs}]')

end_time = time.perf_counter() 
elapsed_time = end_time - start_time

# predições depois do treinamento
model.eval()  # modo de avaliação
with torch.no_grad():
    predictions = model(x_train_tensor)

print(f"Tempo total de treinamento: {elapsed_time:.2f} segundos")

plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, label='Valor Real (Seno)', color='blue')
plt.plot(x_train, predictions.cpu().numpy(), label='Predição pela MLP', color='red', linestyle='--')  # Movendo as predições de volta para a CPU
plt.legend()
plt.title('Aproximação da Função Seno com Rede Neural MLP')
plt.xlabel('Entrada (x)')
plt.ylabel('Saída (y)')
plt.grid(True)
plt.show()