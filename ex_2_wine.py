import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregando o dataset wine
data = load_wine()
X = data.data
y = data.target

# Pré-processamento dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividindo os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.15, random_state=42)

# Função de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(X):
    exp_scores = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# Classe do perceptron
class Perceptron:
    def __init__(self, lr=0.5, epochs=100):
        self.lr = lr
        self.epochs = epochs
        
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        errors = []
        for i in range(self.epochs):
            error = 0
            for xi, yi in zip(X, y):
                y_pred = sigmoid(np.dot(xi, self.weights) + self.bias)
                
                delta_w = self.lr * (yi - y_pred) * xi
                delta_b = self.lr * (yi - y_pred)
  
                self.weights += delta_w                
                self.bias += delta_b
                error += (yi - y_pred) ** 2
            errors.append(error)
        return errors

          

    def predict(self, X):
        y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
        return np.round(y_pred)
    
# Instanciando o perceptron
perceptron = Perceptron(lr=0.1, epochs=100)

# Treinando o perceptron
errors = perceptron.fit(X_train, y_train)

# Predições do modelo
y_pred_train = perceptron.predict(X_train)
y_pred_test = perceptron.predict(X_test)

# Acurácia do modelo
accuracy_train = np.mean(y_pred_train == y_train)
accuracy_test = np.mean(y_pred_test == y_test)

# Plotagem dos resultados
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
ax1.plot(errors)
ax1.set_title('Erro Quadrático')
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Erro')
ax2.scatter(X_train[:,0], X_train[:,1], c=y_train)
ax2.set_title('Treinamento')
ax2.set_xlabel('Característica 1')
ax2.set_ylabel('Característica 2')
ax3.scatter(X_test[:,0], X_test[:,1], c=y_test)
ax3.set_title('Teste')
ax3.set_xlabel('Característica 1')
ax3.set_ylabel('Característica 2')
plt.show()

# Imprimindo os resultados
print('Erro Quadrático:', errors[-1])
print('Acurácia (Treinamento):', accuracy_train)
print('Acurácia (Teste):', accuracy_test)

# softmax results
softmax_scores = softmax(X_test[:10])
#print("Softmax results:")
#print(softmax_scores)

for i in softmax_scores:
    print (np.sum(i))