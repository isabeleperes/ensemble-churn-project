# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 19:20:26 2026

@author: Isabele
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 00:29:15 2026

@author: Isabele
"""

#%% Bibliotecas: as "caixas de ferramentas" que vamos usar
import pandas as pd # trabalhar com tabelas de dados
from sklearn.model_selection import train_test_split # separar dados em treino/teste
from sklearn.tree import DecisionTreeClassifier # o modelo de árvore de decisão
from sklearn.metrics import accuracy_score # medir a proporção de acertos

#%% PASSO 1: Carregar os dados
# read_csv baixa e lê uma tabela (parecida com uma planilha do Excel) direto da internet
url = "https://raw.githubusercontent.com/pplonski/datasets-for-start/refs/heads/master/telco-customer-churn/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
#%% PASSO 2: Dar uma primeira olhada nos dados
print("Formato da tabela (linhas, colunas):", df.shape)
print(df.head()) # mostra as 5 primeiras linhas
df.info() # mostra o nome e o tipo de cada coluna
# Dica: dê um duplo clique em "df" na aba Variable Explorer para ver a tabela
# inteira, como se fosse um Excel.
#%% PASSO 3: Corrigir um problema conhecido deste dataset
# TotalCharges deveria ser número, mas vem como texto, com espaços em branco.
# to_numeric converte; o que não der para converter vira NaN ("não é um número")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna() # remove as poucas linhas com dado ausente
#%% PASSO 4: Separar X (variáveis explicativas) de y (o que queremos prever)
# customerID é só um identificador, não ajuda a prever nada
X = df.drop(columns=['customerID', 'Churn'])
y = df['Churn'] # y = o cliente cancelou ("Yes") ou não ("No")
#%% PASSO 5: Transformar colunas de texto em números (modelos só entendem número)
# get_dummies cria uma coluna de 0/1 para cada categoria de texto
X = pd.get_dummies(X)
#%% PASSO 6: Separar treino (o modelo aprende) e teste (guardado escondido, para conferir)
# test_size=0.2 = 20% vira teste; stratify=y mantém a proporção de Yes/No nos dois grupos
X_treino, X_teste, y_treino, y_teste = train_test_split(
X, y, test_size=0.2, stratify=y, random_state=42
)
#%% PASSO 7: Árvore "podada" manualmente — limitando a profundidade
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
# Comparar com a árvore sem limite do Dia 1
modelo_podada = DecisionTreeClassifier(random_state=42, max_depth=4)
modelo_podada.fit(X_treino, y_treino)
acc_treino_podada = accuracy_score(y_treino,modelo_podada.predict(X_treino))
acc_teste_podada = accuracy_score(y_teste, modelo_podada.predict(X_teste))
print(f"Árvore podada (max_depth=4) — treino: {acc_treino_podada:.2%} |teste: {acc_teste_podada:.2%}")
print(f"Número de folhas: {modelo_podada.get_n_leaves()} (compare com o Dia1)")
#%% PASSO 8: Testar várias combinações de hiperparâmetros automaticamente com GridSearchCV
grade_parametros = {
'max_depth': [3, 5, 7, 10],
'min_samples_leaf': [1, 5, 10, 20],
'criterion': ['gini', 'entropy'],
}
busca = GridSearchCV(
estimator=DecisionTreeClassifier(random_state=42),
param_grid=grade_parametros,
cv=5, # validação cruzada com 5 partições (ver Dia 5)
scoring='accuracy',
n_jobs=-1, # usa todos os núcleos do processador disponíveis
verbose=1,
)
busca.fit(X_treino, y_treino)
print("\nMelhores hiperparâmetros encontrados:", busca.best_params_)
print(f"Melhor acurácia média (validação cruzada): {busca.best_score_:.2%}")
#%% PASSO 9: Avaliar o melhor modelo encontrado no conjunto de teste
melhor_modelo = busca.best_estimator_
predicoes_otimizada = melhor_modelo.predict(X_teste)
acc_teste_otimizada = accuracy_score(y_teste, predicoes_otimizada)
acc_treino_otimizada = accuracy_score(y_treino, modelo_podada.predict(X_treino))
print(f"\nAcurácia no teste (modelo otimizado): {acc_teste_otimizada:.2%}")
print(classification_report(y_teste, predicoes_otimizada))
print(f"\nAcurácia no treino: {acc_teste_otimizada:.2%}")
print(f"\nAcurácia no teste: {acc_treino_otimizada:.2%}")

