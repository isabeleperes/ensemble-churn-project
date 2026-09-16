# -*- coding: utf-8 -*-
"""
Métricas de classificação — relatório de classificação, matriz de confusão
e curva ROC para a árvore de decisão treinada no exercício anterior.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
)
import matplotlib.pyplot as plt

#%% Carregar e preparar os dados (igual ao exercício anterior)
url = "https://raw.githubusercontent.com/pplonski/datasets-for-start/refs/heads/master/telco-customer-churn/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

X = df.drop(columns=['customerID', 'Churn'])
y = df['Churn']
X = pd.get_dummies(X)

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#%% Treinar o modelo
modelo_arvore = DecisionTreeClassifier(random_state=42)
modelo_arvore.fit(X_treino, y_treino)

predicoes_teste = modelo_arvore.predict(X_teste)
predicoes_treino = modelo_arvore.predict(X_treino)

acuracia_treino = accuracy_score(y_treino, predicoes_treino)
acuracia_teste = accuracy_score(y_teste, predicoes_teste)
print(f"\nAcurácia no treino: {acuracia_treino:.2%}")
print(f"Acurácia no teste: {acuracia_teste:.2%}")
print(f"Profundidade da árvore: {modelo_arvore.get_depth()}")
print(f"Número de folhas: {modelo_arvore.get_n_leaves()}")

#%% Relatório de classificação
print("\nRelatório de classificação (teste):")
print(classification_report(y_teste, predicoes_teste))

#%% Matriz de confusão normalizada (percentual por linha)
ConfusionMatrixDisplay.from_estimator(
    modelo_arvore, X_teste, y_teste, normalize='true', cmap='Blues'
)
plt.title('Matriz de Confusão — Decision Tree')
plt.tight_layout()
plt.savefig('images/matriz_confusao_dia2.png')
print("\nGráfico salvo em images/matriz_confusao_dia2.png")

#%% Curva ROC
RocCurveDisplay.from_estimator(
    modelo_arvore, X_teste, y_teste, pos_label='Yes'
)
plt.title('Curva ROC — Decision Tree')
plt.plot([0, 1], [0, 1], '--', color='grey', label='Classificador aleatório')
plt.legend()
plt.tight_layout()
plt.savefig('images/roc_dia2.png')
print("Gráfico salvo em images/roc_dia2.png")
