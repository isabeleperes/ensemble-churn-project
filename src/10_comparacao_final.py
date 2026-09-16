# -*- coding: utf-8 -*-
"""
Comparação final — Decision Tree, Random Forest e XGBoost lado a lado,
com tabela de métricas e curvas ROC sobrepostas.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, RocCurveDisplay,
)

#%% Carregar e preparar os dados
url = "https://raw.githubusercontent.com/pplonski/datasets-for-start/refs/heads/master/telco-customer-churn/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
X = pd.get_dummies(df.drop(columns=['customerID', 'Churn']))
y = (df['Churn'] == 'Yes').astype(int)  # padronizado como número: o XGBoost exige
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#%% Treinar os 3 modelos representativos
modelo_arvore_podada = DecisionTreeClassifier(random_state=42, max_depth=4)
modelo_arvore_podada.fit(X_treino, y_treino)

modelo_rf = RandomForestClassifier(random_state=42, n_estimators=400, min_samples_leaf=5)
modelo_rf.fit(X_treino, y_treino)

modelo_xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=6,
    random_state=42, eval_metric='logloss'
)
modelo_xgb.fit(X_treino, y_treino)

modelos = {
    'Decision Tree (Semana 1)': modelo_arvore_podada,
    'Random Forest (Semana 2)': modelo_rf,
    'XGBoost (Semana 2)': modelo_xgb,
}

#%% Tabela de métricas para os 3 modelos
resultados = []
for nome, modelo in modelos.items():
    y_pred = modelo.predict(X_teste)
    y_prob = modelo.predict_proba(X_teste)[:, 1]  # probabilidade da classe "Yes" (cancelou)
    resultados.append({
        'Modelo': nome,
        'Accuracy': accuracy_score(y_teste, y_pred),
        'Precision': precision_score(y_teste, y_pred),
        'Recall': recall_score(y_teste, y_pred),
        'F1': f1_score(y_teste, y_pred),
        'ROC AUC': roc_auc_score(y_teste, y_prob),
    })
tabela_final = pd.DataFrame(resultados).set_index('Modelo').round(3)
print("\nTabela comparativa final (as 2 semanas juntas):\n")
print(tabela_final)
tabela_final.to_csv('images/tabela_resultados_final.csv')

#%% Curvas ROC dos 3 modelos no mesmo gráfico
fig, eixo = plt.subplots(figsize=(8, 6))
for nome, modelo in modelos.items():
    RocCurveDisplay.from_estimator(modelo, X_teste, y_teste, name=nome, ax=eixo)
eixo.plot([0, 1], [0, 1], '--', color='grey', label='Classificador aleatório')
eixo.set_title('Comparação final — Decision Tree x Random Forest x XGBoost')
eixo.legend(loc='lower right')
plt.tight_layout()
plt.savefig('images/roc_comparacao_final.png')
print("\nGráfico salvo em images/roc_comparacao_final.png")
