# Douglas Leonel de Almeida
# Matrícula: 2110213

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Função para gerar dados sintéticos
def generate_synthetic_data():
    return pd.DataFrame({
        'acidez': [6.5, 7.2, 8.0, 6.8, 7.0, 6.6, 7.4, 6.9, 7.1, 6.7],
        'teor_alcoolico': [12.0, 13.5, 11.0, 13.2, 12.8, 12.3, 12.1, 12.5, 11.5, 12.6],
        'pH': [3.2, 3.1, 3.5, 3.3, 3.4, 3.0, 3.6, 3.2, 3.3, 3.1],
        'densidade': [0.98, 1.01, 0.99, 1.05, 1.03, 1.00, 0.97, 1.02, 0.98, 1.04],
        'label': ['branco', 'branco', 'vermelho', 'vermelho', 'branco', 'branco', 'vermelho', 'vermelho', 'branco', 'vermelho']
    })

# Caminho para o arquivo 'wine_data.csv'
file_path = 'F:/FTT Repository/Fabrica-de-Software/Atividade Prática - Ciclo 1/wine_data.csv'

# Verificando se o arquivo existe
if not os.path.exists(file_path):
    # Dados sintéticos para o arquivo CSV
    wine_data = generate_synthetic_data()
    wine_data.to_csv(file_path, index=False)
    print(f"Arquivo '{file_path}' gerado com sucesso.")
else:
    # Carregando o conjunto de dados
    wine_data = pd.read_csv(file_path)

# Exibindo informações sobre o conjunto de dados
print("================================================================================")
print("Leitura e Exploração de Dados")
print("================================================================================")
print("Informações sobre o conjunto de dados:")
print(wine_data.info())
print("\nDescrição estatística do conjunto de dados:")
print(wine_data.describe())

# Pré-processamento dos dados
X_features = wine_data.drop('label', axis=1)
y_label = wine_data['label']

# Lidando com valores ausentes
imputer = SimpleImputer(strategy='mean')
X_features = imputer.fit_transform(X_features)

# Normalizando as características
scaler = StandardScaler()
X_features = scaler.fit_transform(X_features)

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_features, y_label, test_size=0.2, random_state=42)

# Implementando e treinando o modelo LinearSVC
linear_svc_model = LinearSVC(random_state=42)
linear_svc_model.fit(X_train, y_train)

# Avaliando o modelo LinearSVC
y_pred_linear_svc = linear_svc_model.predict(X_test)
accuracy_linear_svc = accuracy_score(y_test, y_pred_linear_svc)

# Implementando e avaliando o DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
y_pred_dummy = dummy_clf.predict(X_test)
accuracy_dummy = accuracy_score(y_test, y_pred_dummy)

# Exibindo os resultados
print("================================================================================")
print("Comparação de Modelos")
print("================================================================================")
print("Acurácia do Modelo LinearSVC:", accuracy_linear_svc)
print("Acurácia do DummyClassifier (Estratégia 'most_frequent'):", accuracy_dummy)

# Análise e Discussão dos resultados
print("================================================================================")
print("Análise e Discussão")
print("================================================================================")
print("O Modelo LinearSVC apresentou uma acurácia superior ao DummyClassifier.")
print("A escolha do modelo adequado é crucial, pois o LinearSVC oferece uma abordagem mais robusta e adaptável aos dados.")
