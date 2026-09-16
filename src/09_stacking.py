# -*- coding: utf-8 -*-
"""
Stacking — combina Decision Tree, Random Forest e XGBoost num único
meta-modelo (Regressão Logística) treinado sobre as previsões dos três.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score

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

#%% Combinar os 3 modelos num Stacking
# Os modelos de Nível 0 treinam de forma independente; suas previsões viram
# a entrada do meta-modelo de Nível 1 (Regressão Logística), que aprende a
# melhor forma de combiná-las.
modelos_base = [
    ('arvore', DecisionTreeClassifier(random_state=42, max_depth=5)),
    ('rf', RandomForestClassifier(random_state=42, n_estimators=200)),
    ('xgb', XGBClassifier(random_state=42, n_estimators=200, eval_metric='logloss')),
]
modelo_stacking = StackingClassifier(
    estimators=modelos_base,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,  # garante previsões "out-of-fold" para o meta-modelo, evitando vazamento de dados
)
modelo_stacking.fit(X_treino, y_treino)
predicoes_stacking = modelo_stacking.predict(X_teste)
print(f"Acurácia no teste - Stacking: {accuracy_score(y_teste, predicoes_stacking):.2%}")
print(f"Recall no teste - Stacking: {recall_score(y_teste, predicoes_stacking):.2%}")

#%% Quanto os modelos individuais concordam entre si
pred_arvore = modelos_base[0][1].fit(X_treino, y_treino).predict(X_teste)
pred_rf = modelos_base[1][1].fit(X_treino, y_treino).predict(X_teste)
pred_xgb = modelos_base[2][1].fit(X_treino, y_treino).predict(X_teste)

print("Árvore vs RF:", (pred_arvore == pred_rf).mean())
print("Árvore vs XGB:", (pred_arvore == pred_xgb).mean())
print("RF vs XGB:", (pred_rf == pred_xgb).mean())
