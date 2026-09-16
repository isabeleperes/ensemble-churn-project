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
#%% PASSO 7: Criar e treinar o modelo (o "coração" do exercício)
modelo_arvore = DecisionTreeClassifier(random_state=42)
modelo_arvore.fit(X_treino, y_treino) # .fit() = "treine com estes dados"
#%% PASSO 8: Pedir para o modelo prever em dados que ele nunca viu (teste)
predicoes_teste = modelo_arvore.predict(X_teste)
predicoes_treino = modelo_arvore.predict(X_treino)
#%% PASSO 9: Medir o desempenho
acuracia_treino = accuracy_score(y_treino, predicoes_treino)
acuracia_teste = accuracy_score(y_teste, predicoes_teste)
print(f"\nAcurácia no treino: {acuracia_treino:.2%}")
print(f"Acurácia no teste: {acuracia_teste:.2%}")
#%% PASSO 10: Olhar para dentro da árvore — quantos "nós" ela tem, e sua profundidade
print(modelo_arvore.get_n_leaves())
print(f"Profundidade da árvore: {modelo_arvore.get_depth()}")
print(f"Número de folhas: {modelo_arvore.get_n_leaves()}")
#%% PASSO 11: Relatório de classificação
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
# Mostra precisão, recall e F1 por classe de uma vez
print("\nRelatório de classificação (teste):")
print(classification_report(y_teste, predicoes_teste))
#%% PASSO 12: Matriz de confusão normalizada (mostra em % por linha, mais fácil de ler)
ConfusionMatrixDisplay.from_estimator(
modelo_arvore, X_teste, y_teste, normalize='true', cmap='Blues')
plt.title('Matriz de Confusão — Decision Tree')
plt.tight_layout()
plt.savefig('images/matriz_confusao_dia2.png')
print("\nGráfico salvo em images/matriz_confusao_dia2.png — confira também na aba Plots")
#%% PASSO 13: Curva ROC — mostra a capacidade do modelo de distinguir as classes
RocCurveDisplay.from_estimator(
modelo_arvore, X_teste, y_teste, pos_label='Yes'
)
plt.title('Curva ROC — Decision Tree')
plt.plot([0, 1], [0, 1], '--', color='grey', label='Classificador aleatório')
plt.legend()
plt.tight_layout()
plt.savefig('images/roc_dia2.png')
print("Gráfico salvo em images/roc_dia2.png")
