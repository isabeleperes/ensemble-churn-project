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
#%% PASSO 7: Criar e treinar o Random Forest
# Esta linha importa a "receita" de Random Forest do scikit-learn.
# É a mesma biblioteca que trouxe DecisionTreeClassifier na Semana 1 —
# só que agora, em vez de UMA árvore, RandomForestClassifier treina
# VÁRIAS árvores por baixo dos panos e combina o voto de todas.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score,classification_report
import matplotlib.pyplot as plt
# Aqui estamos criando o modelo "vazio", ainda sem ter visto nenhum dado.
# Cada argumento dentro dos parênteses é uma configuração que você escolhe
# ANTES de treinar (são os "hiperparâmetros" que o resumo teórico explica).
modelo_rf = RandomForestClassifier(
random_state=42
# random_state=42: trava a "sorte" do embaralhamento interno do modelo.
# Sem isso, cada vez que você rodasse o código, o Random Forest sortearia
# amostras diferentes para cada árvore, e o resultado final mudaria
# ligeiramente a cada execução. Com random_state fixo, o resultado é
# sempre o mesmo, o que facilita comparar "antes e depois" de uma mudança.
##Repare que NÃO estamos passando n_estimators (número de árvores) nem
# max_features (quantas variáveis cada árvore pode olhar por divisão) —
# isso significa que o scikit-learn vai usar os valores padrão dele:
# 100 árvores, e cada uma olhando a raiz quadrada do total de variáveis
# disponíveis em cada divisão. No Dia 7 vamos testar outros valores.
)
# .fit() = "aprenda com estes dados". Isso é o comando que efetivamente
# treina as 100 árvores por baixo dos panos: cada árvore recebe uma amostra
# "bootstrap" diferente (sorteada com reposição, lembra do resumo?) e um
# subconjunto diferente de variáveis a cada divisão. Depois desse comando,
# o objeto modelo_rf deixa de estar "vazio" e passa a conter as 100 árvores
# já treinadas, prontas para fazer previsões.
modelo_rf.fit(X_treino, y_treino)
# .predict() pede para o modelo "chutar" a resposta (Yes ou No) para cada
# linha que você passar. Aqui estamos pedindo duas previsões separadas:
# uma para os dados que o modelo JÁ VIU (treino) e outra para os dados
# que ele NUNCA VIU (teste) — é a mesma lógica de diagnóstico de
# overfitting que usamos a semana inteira.
predicoes_rf_treino = modelo_rf.predict(X_treino)
predicoes_rf_teste = modelo_rf.predict(X_teste)
# accuracy_score compara duas listas (a resposta certa e a previsão do
# modelo) e devolve a PORCENTAGEM de acertos — um número entre 0 e 1.
# O primeiro argumento é sempre "a verdade" (y_treino ou y_teste),
# o segundo é sempre "o que o modelo chutou" (predicoes_rf_treino/teste).
acuracia_rf_treino = accuracy_score(y_treino, predicoes_rf_treino)
acuracia_rf_teste = accuracy_score(y_teste, predicoes_rf_teste)
# :.2% é um formato especial de texto do Python: ele pega um número tipo 
#0.784 e imprime como "78.40%" — multiplica por 100 e adiciona o símbolo
# de porcentagem automaticamente, para não precisarmos fazer essa conta
# na mão toda vez.
print(f"Acurácia no treino - Random Forest: {acuracia_rf_treino:.2%}")
print(f"Acurácia no teste - Random Forest: {acuracia_rf_teste:.2%}")
# O "gap" (diferença) entre as duas acurácias é o mesmo diagnóstico de
# overfitting que usamos a semana inteira: se o número de baixo for muito
# menor que o de cima, o modelo está "decorando" o treino em vez de
# aprender um padrão que funciona em clientes novos.
gap_rf = (acuracia_rf_treino - acuracia_rf_teste) * 100
print(f"Gap treino-teste: {gap_rf:.1f} pontos percentuais")
#%% PASSO 8: Relatório de classificação completo (precisão, recall, F1)
# Igual ao que fizemos na Semana 1: y_teste é "a verdade", predicoes_rf_teste
# é "o chute do modelo". classification_report calcula, para cada classe
# (0 = não cancelou, 1 = cancelou), a precisão, o recall e o F1-score.
print("\nRelatório de classificação (teste) - Random Forest:")
print(classification_report(y_teste, predicoes_rf_teste))
#%% PASSO 9: Matriz de confusão do Random Forest
# from_estimator já treina a leitura da matriz de confusão automaticamente
# a partir de um modelo já treinado — não precisamos calcular manualmente
# quantos foram verdadeiro-positivo, falso-positivo, etc.
ConfusionMatrixDisplay.from_estimator(
modelo_rf, # o modelo já treinado
X_teste, # os dados de entrada que ele vai prever
y_teste, # a resposta certa, para comparar com a previsão
normalize='true',
# normalize='true' faz o gráfico mostrar PORCENTAGEM por linha, em vez
# de contagem bruta de clientes. Isso facilita comparar visualmente,
# porque a classe "No" tem quase 3x mais clientes que a "Yes" — sem
# normalizar, o quadrado do "No" ficaria visualmente "gigante" e o do
# "Yes" pareceria insignificante, mesmo que as porcentagens de acerto
# sejam parecidas.
cmap='Blues'
# cmap = "color map" = a paleta de cores do gráfico. 'Blues' pinta as
# células mais escuras conforme o valor é mais alto — é só estética,
# não afeta o resultado, só a leitura visual.
)
plt.title('Matriz de Confusão — Random Forest')
plt.tight_layout()
plt.savefig('images/matriz_confusao_rf.png')
print("Gráfico salvo em images/matriz_confusao_rf.png")