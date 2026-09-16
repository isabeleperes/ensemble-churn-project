# -*- coding: utf-8 -*-
"""
Random Forest — primeira floresta aleatória para prever churn, com
avaliação de acurácia, relatório de classificação e matriz de confusão.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt

#%% Carregar e preparar os dados (igual aos exercícios anteriores)
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
X = pd.get_dummies(df.drop(columns=['customerID', 'Churn']))
y = (df['Churn'] == 'Yes').astype(int)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#%% Treinar o Random Forest
# random_state fixa a aleatoriedade interna, para o resultado ser reprodutível.
# n_estimators e max_features ficam no padrão do scikit-learn (100 árvores,
# raiz quadrada do total de variáveis por divisão) — outros valores são
# testados no script de tuning.
modelo_rf = RandomForestClassifier(random_state=42)
modelo_rf.fit(X_treino, y_treino)

predicoes_rf_treino = modelo_rf.predict(X_treino)
predicoes_rf_teste = modelo_rf.predict(X_teste)

acuracia_rf_treino = accuracy_score(y_treino, predicoes_rf_treino)
acuracia_rf_teste = accuracy_score(y_teste, predicoes_rf_teste)
print(f"Acurácia no treino - Random Forest: {acuracia_rf_treino:.2%}")
print(f"Acurácia no teste - Random Forest: {acuracia_rf_teste:.2%}")

gap_rf = (acuracia_rf_treino - acuracia_rf_teste) * 100
print(f"Gap treino-teste: {gap_rf:.1f} pontos percentuais")

#%% Relatório de classificação
print("\nRelatório de classificação (teste) - Random Forest:")
print(classification_report(y_teste, predicoes_rf_teste))

#%% Matriz de confusão (normalizada por linha, já que "No" tem ~3x mais clientes que "Yes")
ConfusionMatrixDisplay.from_estimator(
    modelo_rf, X_teste, y_teste, normalize='true', cmap='Blues'
)
plt.title('Matriz de Confusão — Random Forest')
plt.tight_layout()
plt.savefig('images/matriz_confusao_rf.png')
print("Gráfico salvo em images/matriz_confusao_rf.png")
