# -*- coding: utf-8 -*-
"""
Primeira árvore de decisão — previsão de churn (cancelamento de clientes)
usando o dataset Telco Customer Churn.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#%% Carregar os dados
url = "https://raw.githubusercontent.com/pplonski/datasets-for-start/refs/heads/master/telco-customer-churn/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print("Formato da tabela (linhas, colunas):", df.shape)
print(df.head())
df.info()

#%% Tratamento: TotalCharges vem como texto, com espaços em branco
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

#%% Separar variáveis explicativas (X) do alvo (y)
X = df.drop(columns=['customerID', 'Churn'])  # customerID não ajuda a prever nada
y = df['Churn']

#%% Codificar variáveis categóricas em dummies (0/1)
X = pd.get_dummies(X)

#%% Separar treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#%% Treinar a árvore de decisão
modelo_arvore = DecisionTreeClassifier(random_state=42)
modelo_arvore.fit(X_treino, y_treino)

#%% Avaliar desempenho
predicoes_teste = modelo_arvore.predict(X_teste)
predicoes_treino = modelo_arvore.predict(X_treino)

acuracia_treino = accuracy_score(y_treino, predicoes_treino)
acuracia_teste = accuracy_score(y_teste, predicoes_teste)
print(f"\nAcurácia no treino: {acuracia_treino:.2%}")
print(f"Acurácia no teste: {acuracia_teste:.2%}")

#%% Tamanho da árvore
print(f"Número de folhas: {modelo_arvore.get_n_leaves()}")
print(f"Profundidade da árvore: {modelo_arvore.get_depth()}")
