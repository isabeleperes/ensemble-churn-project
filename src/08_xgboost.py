# -*- coding: utf-8 -*-
"""
XGBoost — modelo de boosting sequencial para prever churn, com avaliação
e importância de variáveis.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

#%% Carregar e preparar os dados (igual aos exercícios anteriores)
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
X = pd.get_dummies(df.drop(columns=['customerID', 'Churn']))
y = (df['Churn'] == 'Yes').astype(int)  # XGBoost exige alvo numérico (0/1), não texto
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#%% Treinar o XGBoost
# Diferente do Random Forest (árvores em paralelo/bagging), o XGBoost treina
# árvores em sequência, cada uma corrigindo o erro da anterior (boosting).
modelo_xgb = XGBClassifier(
    n_estimators=300,      # número máximo de árvores na sequência
    learning_rate=0.1,     # peso de cada correção — baixo = aprendizado mais estável
    max_depth=6,           # profundidade de cada árvore individual
    random_state=42,
    eval_metric='logloss',
)
modelo_xgb.fit(X_treino, y_treino)

predicoes_xgb_treino = modelo_xgb.predict(X_treino)
predicoes_xgb_teste = modelo_xgb.predict(X_teste)
acc_treino_xgb = accuracy_score(y_treino, predicoes_xgb_treino)
acc_teste_xgb = accuracy_score(y_teste, predicoes_xgb_teste)
print(f"Acurácia no treino - XGBoost: {acc_treino_xgb:.2%}")
print(f"Acurácia no teste - XGBoost: {acc_teste_xgb:.2%}")
print(f"Gap treino-teste: {(acc_treino_xgb - acc_teste_xgb)*100:.1f} pontos percentuais")

print("\nRelatório de classificação (teste) - XGBoost:")
print(classification_report(y_teste, predicoes_xgb_teste))

#%% Importância de variáveis
importancias_xgb = pd.Series(
    modelo_xgb.feature_importances_,
    index=X_treino.columns
).sort_values(ascending=False)
print("\nTop 10 variáveis mais importantes - XGBoost:")
print(importancias_xgb.head(10))
