# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:13:22 2026

@author: Isabele
"""

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
#%% PASSO 7: Validação cruzada K-fold "de verdade", com 5 partições estratificadas
# StratifiedKFold garante que a proporção de 0/1 fique parecida em cada partição
validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
modelo_arvore_padrao = DecisionTreeClassifier(random_state=42)
scores_cv = cross_val_score(modelo_arvore_padrao, X, y, cv=validacao,scoring='accuracy')
print("Resultado da validação cruzada (árvore padrão):")
print(f"Acurácia média: {scores_cv.mean():.2%} | Desvio-padrão:{scores_cv.std():.2%}")
#%% PASSO 8: Treinar os 3 modelos da semana e comparar lado a lado
modelo_padrao = DecisionTreeClassifier(random_state=42)
modelo_padrao.fit(X_treino, y_treino)
modelo_podada = DecisionTreeClassifier(random_state=42, max_depth=4)
modelo_podada.fit(X_treino, y_treino)
grade_parametros = {'max_depth': [3, 5, 7, 10], 'min_samples_leaf': [1, 5,10, 20]}
busca = GridSearchCV(
DecisionTreeClassifier(random_state=42), grade_parametros, cv=5,scoring='accuracy', n_jobs=-1)
busca.fit(X_treino, y_treino)
print(busca.best_estimator_)
print(busca.best_params_)
modelo_otimizada = busca.best_estimator_
modelos = {
'Decision Tree (padrão)': modelo_padrao,
'Decision Tree (podada)': modelo_podada,
'Decision Tree (otimizada)': modelo_otimizada,
}
#%% PASSO 9: Montar a tabela final da Semana 1 — guarde esse resultado, vamos usá-lo na Semana 2!
resultados = []
for nome, modelo in modelos.items():
    y_pred = modelo.predict(X_teste)
    y_prob = modelo.predict_proba(X_teste)[:, 1]
    resultados.append({
        'Modelo': nome,
        'Accuracy': accuracy_score(y_teste, y_pred),
        'Precision': precision_score(y_teste, y_pred),
        'Recall': recall_score(y_teste, y_pred),
        'F1': f1_score(y_teste, y_pred),
        'ROC AUC': roc_auc_score(y_teste, y_prob),
    })
tabela_semana1 = pd.DataFrame(resultados).set_index('Modelo').round(3)
print("\nTabela final da Semana 1:\n")
print(tabela_semana1)
tabela_semana1.to_csv('images/tabela_resultados_semana1.csv')
print("\nTabela salva em images/tabela_resultados_semana1.csv — guarde este arquivo!")

#%% Plotando as 3 tabelas juntas para avaliar os resultados
for nome, modelo in modelos.items():
    y_pred = modelo.predict(X_teste)
    print(f"\n{'='*50}")
    print(f"Relatório — {nome}")
    print('='*50)
    print(classification_report(y_teste, y_pred))
