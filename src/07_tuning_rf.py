# -*- coding: utf-8 -*-
"""
Tuning do Random Forest com GridSearchCV, otimizando para recall, e
análise das variáveis mais importantes do modelo.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

#%% Grade de hiperparâmetros a testar
grade_parametros = {
    'n_estimators': [200, 400],
    'max_depth': [None, 10],
    'min_samples_leaf': [1, 5],
}
# Total de combinações: 2 x 2 x 2 = 8, cada uma avaliada com 5-fold cross-validation
# (40 treinos no total). scoring='recall' porque, em churn, identificar quem vai
# cancelar importa mais do que só acertar a maioria (ver conclusão da Semana 1).
busca = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=grade_parametros,
    cv=5,
    scoring='recall',
    n_jobs=-1,
    verbose=1,
)
busca.fit(X_treino, y_treino)

print("\nMelhores parâmetros encontrados:", busca.best_params_)
print(f"Melhor recall médio (validação cruzada): {busca.best_score_:.2%}")

melhor_modelo = busca.best_estimator_
predicoes_teste = melhor_modelo.predict(X_teste)
print(f"Acurácia no teste com o melhor modelo: {accuracy_score(y_teste, predicoes_teste):.2%}")

#%% Variáveis mais importantes segundo o modelo otimizado
importancias = pd.Series(
    melhor_modelo.feature_importances_,
    index=X_treino.columns
).sort_values(ascending=False)
top_10 = importancias.head(10).sort_values()  # crescente, para o gráfico horizontal

plt.figure(figsize=(10, 6))
top_10.plot(kind='barh', color='steelblue')
plt.title('Top 10 variáveis mais importantes — Random Forest ajustada')
plt.xlabel('Importância')
plt.tight_layout()
plt.savefig('images/feature_importance_rf.png')
print("Gráfico salvo em images/feature_importance_rf.png")
