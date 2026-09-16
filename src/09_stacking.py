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
#%% PASSO 7: Combinar Decision Tree, Random Forest e XGBoost num Stacking
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score
# modelos_base é uma LISTA de pares (apelido, modelo) — cada modelo aqui
# é chamado de "Nível 0" no stacking: eles vão treinar de forma
# independente, e as previsões de CADA UM viram uma "pista" de entrada
# para o modelo final (Nível 1), lá embaixo.
modelos_base = [
('arvore', DecisionTreeClassifier(random_state=42, max_depth=5)),
# 'arvore' é só um apelido texto para identificar esse modelo dentro
# do stacking — pode ser qualquer nome, é só um rótulo interno.
('rf', RandomForestClassifier(random_state=42, n_estimators=200)),
('xgb', XGBClassifier(random_state=42, n_estimators=200,eval_metric='logloss')),
]
modelo_stacking = StackingClassifier(
estimators=modelos_base,
# a lista de modelos de Nível 0 que acabamos de montar.
final_estimator=LogisticRegression(max_iter=1000),
# final_estimator é o modelo de Nível 1 (o "meta-modelo") — ele NÃO
# olha para os dados originais (tempo de casa, gasto mensal, etc.),
# ele olha só para as PREVISÕES dos três modelos de Nível 0 e aprende
# a melhor forma de combiná-las. Aqui usamos uma Regressão Logística,
# um modelo simples e rápido, comum para esse papel de "juiz final".
# max_iter=1000 é apenas um limite técnico de quantas tentativas
# internas a Regressão Logística pode fazer para convergir numa
# resposta — só evita um aviso de erro caso o padrão (100) não seja
# suficiente para este conjunto de dados.
cv=5,
# este cv=5 cumpre um papel especial aqui: ele garante que as
# previsões que alimentam o meta-modelo sejam "out-of-fold" — ou
# seja, nenhum modelo de Nível 0 nunca vê, ao gerar a previsão que
# alimenta o meta-modelo, os mesmos dados que ele usou para treinar
# (evitando o vazamento de dados que o resumo teórico menciona na
# Parte B.6). Por baixo dos panos, o StackingClassifier já faz essa
# divisão em 5 partições automaticamente, sem você precisar programar
# isso manualmente.
)
modelo_stacking.fit(X_treino, y_treino)
predicoes_stacking = modelo_stacking.predict(X_teste)
print(f"Acurácia no teste - Stacking: {accuracy_score(y_teste,predicoes_stacking):.2%}")
print(f"Recall no teste - Stacking: {recall_score(y_teste,predicoes_stacking):.2%}")

pred_arvore = modelos_base[0][1].fit(X_treino, y_treino).predict(X_teste)
pred_rf = modelos_base[1][1].fit(X_treino, y_treino).predict(X_teste)
pred_xgb = modelo_xgb.predict(X_teste)

# quanto as previsões dos 3 modelos concordam entre si (em %)
print("Árvore vs RF:", (pred_arvore == pred_rf).mean())
print("Árvore vs XGB:", (pred_arvore == pred_xgb).mean())
print("RF vs XGB:", (pred_rf == pred_xgb).mean())