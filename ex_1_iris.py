import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Carregando o dataset
data = load_iris()
X = data.data
y = data.target

# Pré-processamento dos dados atravéz do zscore
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separa os dados em treinamento, validação e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

# Função de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Classe do perceptron
class Perceptron:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
    
    #Treinamento da rede    
    def fit(self, X, y,X_val=None, y_val=None):
        # Inicializa os pesos com um array de zeros
        self.weights = np.zeros(X.shape[1])
        # Inicializa o bias com 0
        self.bias = 0
        # inicia as variaveis de validação
        self.X_val = X_val
        self.y_val = y_val

        val_errors = [] 
        errors = []
        for i in range(self.epochs):
            error = 0
            for xi, yi in zip(X, y):
                y_pred = sigmoid(np.dot(xi, self.weights) + self.bias)
                
                #Variavéis para depuração da atualização de pesos
                delta_w = self.lr * (yi - y_pred) * xi
                delta_b = self.lr * (yi - y_pred)
                #Atualização dos pesos
                self.weights += delta_w                
                self.bias += delta_b
                error += (yi - y_pred) ** 2
            errors.append(error)

            # Cálculo do erro de validação
            if X_val is not None and y_val is not None:
                val_error = 0
                for xi, yi in zip(X_val, y_val):
                    y_val_pred = sigmoid(np.dot(xi, self.weights) + self.bias)
                    val_error += (yi - y_val_pred) ** 2
                val_errors.append(val_error)

        return errors, val_errors

          
    # Função para predição dos resultados
    def predict(self, X):
        y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
        return np.round(y_pred)
    
# Instanciando o perceptron (lr = Taxa de aprendizado e Epochs = Iterações )
perceptron = Perceptron(lr=0.1, epochs=100)

# Treinando o perceptron
errors, val_errors = perceptron.fit(X_train, y_train, X_val, y_val)

# Predições do modelo
y_pred_train = perceptron.predict(X_train)
y_pred_test = perceptron.predict(X_test)

# Acurácia do modelo
accuracy_train = np.mean(y_pred_train == y_train)
accuracy_test = np.mean(y_pred_test == y_test)

# Plotagem dos resultados
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
ax1.plot(errors)
ax1.set_title('Erro Quadrático de treinamento')
ax1.set_xlabel('Épocas')
ax1.set_ylabel('Erro')
ax2.scatter(X_train[:,0], X_train[:,2], c=y_train)
ax2.set_title('Treinamento')
ax3.scatter(X_test[:,0], X_test[:,2], c=y_test)
ax3.set_title('Predição')
ax4.plot(val_errors)
ax4.set_title('Erro Quadrático de validação')
ax4.set_xlabel('Épocas')
ax4.set_ylabel('Erro')
plt.show()

# Imprimindo os resultados
print('Erro Quadrático:', errors[-1])
print('Acurácia (Treinamento):', accuracy_train)
print('Acurácia (Teste):', accuracy_test)

#Matriz de Confusão
cmteste = confusion_matrix(y_test, y_pred_test)

# Extrair os valores da matriz de confusão
tn = cmteste[0, 0]   # Verdadeiros Negativos (classe 0)
fp = cmteste[0, 1] + cmteste[1, 0] + cmteste[1, 2] # Falsos Positivos (classe 1)
fn = cmteste[0, 2] + cmteste[2, 0]  + cmteste[2, 1] # Falsos Negativos (classe 2)
tp = cmteste[1, 1] + cmteste[2, 2] # Verdadeiros Positivos (classe 1 e 2)

# Imprimir os valores
print("Matriz de Confusão de teste x Validação:")
print("Verdadeiros Positivos:", tp)
print("Verdadeiros Negativos:", tn)
print("Falsos Positivos:", fp)
print("Falsos Negativos:", fn)

# Plota a matriz de confusão do teste X
plt.imshow(cmteste, cmap='Blues')
plt.title('Matriz de confusão x validação')
plt.colorbar()
plt.xlabel('Predição')
plt.ylabel('Real')
plt.xticks(np.arange(len(data.target_names)), data.target_names)
plt.yticks(np.arange(len(data.target_names)), data.target_names)

# Plotagem das dimensões da matriz de confusão
plt.text(0, 0, cmteste[0, 0], ha='center', va='center') 
plt.text(0, 1, cmteste[0, 1], ha='center', va='center')
plt.text(0, 2, cmteste[0, 2], ha='center', va='center')  
plt.text(1, 1, cmteste[1, 1], ha='center', va='center') 
plt.text(1, 0, cmteste[1, 0], ha='center', va='center')
plt.text(1, 2, cmteste[1, 2], ha='center', va='center')  
plt.text(2, 2, cmteste[2, 2], ha='center', va='center') 
plt.text(2, 0, cmteste[2, 0], ha='center', va='center')
plt.text(2, 1, cmteste[2, 1], ha='center', va='center')

plt.show()