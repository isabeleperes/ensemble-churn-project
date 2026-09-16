# -*- coding: utf-8 -*-
"""
Árvore de decisão para regressão — previsão de preço de imóveis usando
o dataset California Housing (scikit-learn).
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#%% Carregar o dataset (preços de imóveis na Califórnia, em unidades de $100.000)
dados = fetch_california_housing(as_frame=True)
X = dados.data  # variáveis explicativas (renda média da região, idade do imóvel, etc.)
y = dados.target  # preço médio do imóvel

print("Formato de X:", X.shape)
print(X.head())
print("\nPrimeiros valores de y (preço em unidades de $100 mil):")
print(y.head())

#%% Separar treino e teste (sem stratify: y é numérico, não categoria)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#%% Treinar a árvore de regressão
modelo_regressao = DecisionTreeRegressor(random_state=42)
modelo_regressao.fit(X_treino, y_treino)
predicoes_treino = modelo_regressao.predict(X_treino)
predicoes_teste = modelo_regressao.predict(X_teste)

#%% Métricas de regressão
mae = mean_absolute_error(y_teste, predicoes_teste)
mse = mean_squared_error(y_teste, predicoes_teste)
rmse = np.sqrt(mse)
r2 = r2_score(y_teste, predicoes_teste)
print(f"\nMAE (Erro Médio Absoluto): {mae:.3f}")
print(f"MSE (Erro Quadrático Médio): {mse:.3f}")
print(f"RMSE (Raiz do Erro Quadrático): {rmse:.3f}")
print(f"R² (Coeficiente de Determinação): {r2:.3f}")

#%% Comparar com o desempenho no treino (diagnóstico de overfitting)
r2_treino = r2_score(y_treino, predicoes_treino)
print(f"\nR² no treino: {r2_treino:.3f} | R² no teste: {r2:.3f}")
