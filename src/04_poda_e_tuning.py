# -*- coding: utf-8 -*-
"""
Poda de árvore e tuning de hiperparâmetros com GridSearchCV.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

#%% Carregar e preparar os dados (igual aos exercícios anteriores)
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

#%% Árvore podada manualmente (comparar com a árvore sem limite)
modelo_podada = DecisionTreeClassifier(random_state=42, max_depth=4)
modelo_podada.fit(X_treino, y_treino)
acc_treino_podada = accuracy_score(y_treino, modelo_podada.predict(X_treino))
acc_teste_podada = accuracy_score(y_teste, modelo_podada.predict(X_teste))
print(f"Árvore podada (max_depth=4) — treino: {acc_treino_podada:.2%} | teste: {acc_teste_podada:.2%}")
print(f"Número de folhas: {modelo_podada.get_n_leaves()}")

#%% Busca de hiperparâmetros com GridSearchCV
grade_parametros = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [1, 5, 10, 20],
    'criterion': ['gini', 'entropy'],
}
busca = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=grade_parametros,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
)
busca.fit(X_treino, y_treino)
print("\nMelhores hiperparâmetros encontrados:", busca.best_params_)
print(f"Melhor acurácia média (validação cruzada): {busca.best_score_:.2%}")

#%% Avaliar o melhor modelo encontrado no conjunto de teste
melhor_modelo = busca.best_estimator_
predicoes_otimizada = melhor_modelo.predict(X_teste)
acc_teste_otimizada = accuracy_score(y_teste, predicoes_otimizada)
acc_treino_otimizada = accuracy_score(y_treino, modelo_podada.predict(X_treino))
print(f"\nAcurácia no teste (modelo otimizado): {acc_teste_otimizada:.2%}")
print(classification_report(y_teste, predicoes_otimizada))
print(f"\nAcurácia no treino: {acc_treino_otimizada:.2%}")
print(f"\nAcurácia no teste: {acc_teste_otimizada:.2%}")
