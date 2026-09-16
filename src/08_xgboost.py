# -*- coding: utf-8 -*-

#%% Bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold,cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,classification_report
)
#%% PASSOS 1 a 6: carregar e preparar os dados (igual aos dias anteriores)
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
X = pd.get_dummies(df.drop(columns=['customerID', 'Churn']))
y = (df['Churn'] == 'Yes').astype(int) # 1 = cancelou, 0 = não cancelou
X_treino, X_teste, y_treino, y_teste = train_test_split(
X, y, test_size=0.2, stratify=y, random_state=42
)
#%% PASSO 7: Criar e treinar o XGBoost
# XGBClassifier vem de uma biblioteca separada (xgboost), não do
# scikit-learn — por isso o import é diferente dos anteriores. Mas a
# interface (.fit, .predict) é praticamente idêntica, de propósito, para
# ser fácil de trocar de modelo sem reescrever todo o resto do código.
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
modelo_xgb = XGBClassifier(
n_estimators=300,
# n_estimators aqui significa algo DIFERENTE do Random Forest: não são
# árvores treinadas em paralelo e depois combinadas por votação, são
# árvores treinadas EM SEQUÊNCIA, cada uma corrigindo o erro da
# anterior (é o "boosting" que o resumo teórico explica). 300 é o
# número máximo dessa sequência.
learning_rate=0.1,
# learning_rate controla o "tamanho do passo" de cada correção. Um
# valor baixo como 0.1 significa que cada nova árvore da sequência só
# contribui com 10% da correção que ela "gostaria" de fazer — isso
# torna o aprendizado mais lento, mas mais estável e menos propenso a
# "passar do ponto" (overfitting rápido). Se fosse 1.0, cada árvore
# tentaria corrigir o erro inteiro de uma vez, arriscando oscilar.
max_depth=6,
# max_depth aqui é a profundidade de CADA árvore individual dentro da
# sequência de 300 — diferente do max_depth do Random Forest, que
# também limita a árvore, mas dentro de uma lógica de bagging, não
# de correção sequencial.
random_state=42,
# mesma função de sempre: trava a aleatoriedade interna para o
# resultado ser reproduzível.
eval_metric='logloss',
# eval_metric é a métrica interna que o XGBoost usa, durante o
# próprio treinamento, para saber "o quão errado" ele está a cada
# passo e decidir como ajustar a próxima árvore da sequência.
# 'logloss' é uma métrica pensada para classificação, que penaliza
# bastante previsões erradas feitas "com muita confiança" — não é a
# mesma coisa que a acurácia ou o F1 que usamos para AVALIAR o modelo
# depois de pronto; é uma métrica de "orientação" durante o treino.
)
# O XGBoost, diferente do scikit-learn puro, espera que a variável alvo
# (y) seja NUMÉRICA (0 ou 1), não texto ("Yes"/"No"). Esta linha converte:
# a expressão (y_treino == 'Yes') cria uma lista de True/False (True onde
# o cliente cancelou), e .astype(int) transforma True em 1 e False em 0.

modelo_xgb.fit(X_treino, y_treino)
# Note que aqui usamos y_treino_num (a versão numérica), não y_treino
# (a versão em texto) — se você esquecer essa conversão, o XGBoost pode
# dar erro ou se comportar de forma inesperada.
predicoes_xgb_treino = modelo_xgb.predict(X_treino)
predicoes_xgb_teste = modelo_xgb.predict(X_teste)
acc_treino_xgb = accuracy_score(y_treino, predicoes_xgb_treino)
acc_teste_xgb = accuracy_score(y_teste, predicoes_xgb_teste)
print(f"Acurácia no treino - XGBoost: {acc_treino_xgb:.2%}")
print(f"Acurácia no teste - XGBoost: {acc_teste_xgb:.2%}")
print(f"Gap treino-teste: {(acc_treino_xgb - acc_teste_xgb)*100:.1f} pontos percentuais")
print("\nRelatório de classificação (teste) - XGBoost:")
# Aqui usamos y_teste_num (0/1) em vez de y_teste (Yes/No) porque as
# previsões do XGBoost também saem como 0/1 — as duas listas comparadas
# pelo classification_report precisam estar no mesmo "formato".
print(classification_report(y_teste, predicoes_xgb_teste))
#%% PASSO 8: Importância de variáveis do XGBoost
# Mesma lógica do Random Forest: cada variável recebe uma pontuação de
# o quanto ela ajudou o modelo a separar as classes, ao longo de todas
# as 300 árvores da sequência.
importancias_xgb = pd.Series(
modelo_xgb.feature_importances_,
index=X_treino.columns
).sort_values(ascending=False)
print("\nTop 10 variáveis mais importantes - XGBoost:")
print(importancias_xgb.head(10))
