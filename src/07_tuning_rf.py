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
#%% PASSO 7: Testar várias configurações de Random Forest automaticamente
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
# Este dicionário é a "lista de compras" de configurações que queremos
# testar. Cada chave é o nome de um hiperparâmetro; cada valor é uma
# LISTA de opções que queremos experimentar para aquele hiperparâmetro.
grade_parametros = {
'n_estimators': [200, 400],
# n_estimators = quantas árvores compõem a floresta. Vamos testar uma
# floresta com 200 árvores e outra com 400, para ver se "mais árvores"
# realmente compensa o tempo extra de treino.
'max_depth': [None, 10],
# max_depth = profundidade máxima de CADA árvore dentro da floresta.
# None (o valor especial do Python para "nenhum") significa "sem
# limite" — cada árvore pode crescer até separar tudo, igual à árvore
# do Dia 1. 10 significa "pare em 10 níveis", uma poda mais suave.
'min_samples_leaf': [1, 5],
# min_samples_leaf = quantos clientes, no mínimo, cada "folha" final
# de cada árvore precisa ter. 1 permite folhas bem específicas
# (potencialmente decorando casos raros); 5 exige generalização maior.
}
# GridSearchCV é o "robô" que vai treinar UM modelo para CADA combinação
# possível dentro da grade acima, e escolher a melhor no final.
busca = GridSearchCV(
estimator=RandomForestClassifier(random_state=42),
# estimator = o "molde" de modelo que será testado repetidamente.
# Repare que aqui você só passa random_state — os outros hiperparâmetros
# (n_estimators, max_depth, min_samples_leaf) NÃO vão aqui, porque
# é justamente isso que o param_grid abaixo vai preencher, testando
# cada combinação.
param_grid=grade_parametros,
# a grade de combinações que acabamos de montar.
# Total de combinações = 2 (n_estimators) × 2 (max_depth) × 2 (min_samples_leaf) = 8.
cv=5,
# cv=5 quer dizer: para CADA uma das 8 combinações, não testamos só
# uma vez — dividimos os dados de treino em 5 pedaços (chamados
# "folds"), e repetimos o treino 5 vezes, cada vez usando 4 pedaços
# para aprender e 1 pedaço diferente (que o modelo nunca viu naquela
# rodada) para conferir. No final, a nota daquela combinação é a MÉDIA
# dessas 5 rodadas — isso deixa a comparação entre combinações mais
# confiável do que testar cada uma só uma vez com sorte/azar de sorteio.
# Total de treinos reais = 8 combinações × 5 partições = 40 treinos.
scoring='recall',
# scoring define qual métrica o GridSearchCV usa para decidir "qual
# combinação é a melhor". Aqui escolhemos recall, e não acurácia,
# de propósito: como você mesmo concluiu na tarefa da Semana 1, em
# churn normalmente vale mais identificar quem vai cancelar (recall
# alto) do que só acertar a maioria (acurácia alta). Se você trocasse
# para scoring='accuracy', o robô escolheria uma configuração
# diferente, otimizada para outro objetivo.
n_jobs=-1,
# n_jobs = quantos "trabalhadores" (núcleos do processador) rodam os
# 40 treinos em paralelo, ao mesmo tempo, em vez de um de cada vez.
# -1 é um código especial que significa "use TODOS os núcleos
# disponíveis no computador" — deixa o processo mais rápido.
verbose=1,
# verbose controla o quanto de informação aparece no Console IPython
# enquanto o robô trabalha. verbose=1 mostra uma linha de progresso
# tipo "Fitting 5 folds for each of 8 candidates, totalling 40 fits".
# Se colocasse verbose=0, rodaria em silêncio, sem mostrar nada até
# terminar.
)
# .fit() aqui dispara TODOS os 40 treinos de uma vez. Isso pode demorar
# de alguns segundos a poucos minutos, dependendo do computador.
busca.fit(X_treino, y_treino)

# Depois que o robô termina, ele guarda os resultados dentro do próprio
# objeto "busca". best_params_ é a combinação (das 8) que teve a MAIOR
# nota média de recall nas 5 partições.
print("\nMelhores parâmetros encontrados:", busca.best_params_)
# best_score_ é o valor dessa nota — lembre que essa nota vem só dos
# dados de TREINO (dividido internamente em 5), sem nunca espiar o
# X_teste/y_teste que guardamos lá no início. Isso é importante: o robô
# escolhe a melhor configuração "no escuro", sem trapacear olhando o teste.
print(f"Melhor recall médio (validação cruzada): {busca.best_score_:.2%}")
# best_estimator_ é o modelo JÁ TREINADO com a melhor combinação
# encontrada — pronto para usar diretamente, sem precisar retreinar.
melhor_modelo = busca.best_estimator_
predicoes_teste = melhor_modelo.predict(X_teste)
print(f"Acurácia no teste com o melhor modelo: {accuracy_score(y_teste, predicoes_teste):.2%}")
#%% PASSO 8: Quais variáveis o modelo considerou mais importantes?
# feature_importances_ é um atributo especial que o Random Forest guarda
# depois de treinado: para CADA coluna de entrada (cada variável do
# X_treino), ele dá uma pontuação de 0 a 1 dizendo o quanto aquela
# variável ajudou, em média, a separar as classes em todas as 200/400
# árvores da floresta. A soma de todas as pontuações dá 1 (100%).
importancias = pd.Series(
melhor_modelo.feature_importances_,
# os números de importância, um por variável, na mesma ordem das colunas
index=X_treino.columns
# index = os nomes das variáveis, para conseguirmos ler qual pontuação
# pertence a qual coluna (em vez de só uma lista de números soltos)
).sort_values(ascending=False)
# sort_values(ascending=False) reordena da MAIOR importância para a MENOR,
# para as variáveis mais relevantes aparecerem primeiro.
top_10 = importancias.head(10).sort_values()
# .head(10) pega só as 10 primeiras (as mais importantes).
# O segundo .sort_values() (sem ascending=False, ou seja, crescente) é só
# para o gráfico de barras ficar com a maior importância no topo visual
# quando desenhado na horizontal — é um ajuste estético, não muda os dados.
plt.figure(figsize=(10, 6))
# figsize define o tamanho da figura em polegadas (largura, altura) —
# só controla o tamanho da imagem gerada.
top_10.plot(kind='barh', color='steelblue')
# kind='barh' = "bar horizontal", um gráfico de barras deitadas (mais
# fácil de ler nomes compridos de variável do que barras verticais).
plt.title('Top 10 variáveis mais importantes — Random Forest ajustada')
plt.xlabel('Importância')
plt.tight_layout()
# tight_layout ajusta as margens automaticamente para nada ficar cortado
# fora da imagem (títulos, rótulos dos eixos, etc.)
plt.savefig('images/feature_importance_rf.png')
print("Gráfico salvo em images/feature_importance_rf.png")