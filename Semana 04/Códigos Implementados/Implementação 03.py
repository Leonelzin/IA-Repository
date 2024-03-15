#Aluno: Douglas Leonel de Almeida
#Matrícula: 2110213

# Você foi contratado pelo Banco Bradesco para criar um sistema que diga se o banco deve emprestar dinheiro para uma pessoa ou não. 
#Pense nas características que são relevantes para decidir se uma pessoa deve ou não receber o emprestimo. 
#Crie ao menos 20 entradas diferentes para o treinamento. Crie o modelo de machine learning utlizando o LinearSVC. 
#Valide o modelo e calcule a acúrária utilizando ao menos 7 entradas. Por fim compare a acurácia obtida neste modelo com o Classificador não guiado (DummyClassifier).

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

# Características relevantes para decisão de empréstimo
# As características serão representadas por valores binários (0 ou 1)

# Dados fictícios de treinamento
X_train = np.array([
    [1, 1, 1, 1, 0, 0, 0, 1],  # Exemplo 1
    [0, 1, 0, 1, 1, 0, 1, 0],  # Exemplo 2
    [0, 0, 1, 1, 1, 1, 0, 0],  # Exemplo 3
    [1, 0, 0, 1, 0, 1, 1, 1],  # Exemplo 4
    [1, 1, 1, 0, 0, 0, 1, 0],  # Exemplo 5
    [0, 1, 0, 1, 1, 0, 0, 1],  # Exemplo 6
    [1, 0, 1, 0, 1, 0, 1, 0],  # Exemplo 7
    [0, 1, 1, 0, 0, 1, 1, 0],  # Exemplo 8
    [1, 1, 0, 1, 0, 1, 0, 1],  # Exemplo 9
    [0, 0, 1, 0, 1, 1, 1, 0],  # Exemplo 10
    [1, 0, 0, 1, 0, 1, 1, 0],  # Exemplo 11
    [0, 1, 1, 0, 1, 0, 0, 1],  # Exemplo 12
    [1, 1, 0, 1, 0, 0, 1, 1],  # Exemplo 13
    [0, 1, 0, 1, 1, 1, 0, 1],  # Exemplo 14
    [1, 0, 1, 0, 0, 1, 1, 0],  # Exemplo 15
    [0, 0, 1, 1, 1, 0, 0, 1],  # Exemplo 16
    [1, 1, 0, 0, 1, 1, 0, 1],  # Exemplo 17
    [1, 0, 1, 0, 0, 0, 1, 0],  # Exemplo 18
    [0, 1, 1, 1, 1, 0, 0, 1],  # Exemplo 19
    [0, 0, 0, 1, 0, 1, 1, 0]   # Exemplo 20
])

# Rótulos (0 = Não emprestar, 1 = Emprestar)
y_train = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0])

# Separar conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Treinar modelo LinearSVC
model = LinearSVC()
model.fit(X_train, y_train)

# Validar o modelo
y_pred = model.predict(X_test)

# Calcular acurácia
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo LinearSVC:", accuracy)

# Comparar com DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_pred = dummy_clf.predict(X_test)
dummy_accuracy = accuracy_score(y_test, dummy_pred)
print("Acurácia do DummyClassifier:", dummy_accuracy)
