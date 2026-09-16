# -*- coding: utf-8 -*-

#%% Bibliotecas
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
#%% PASSOS 1-6: carregar e preparar os dados
url = "https://raw.githubusercontent.com/pplonski/datasets-for-start/refs/heads/master/telco-customer-churn/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
X = pd.get_dummies(df.drop(columns=['customerID', 'Churn']))
y = (df['Churn'] == 'Yes').astype(int)
# Aqui já convertemos y para 0/1 direto na criação, para servir aos três
# modelos igualmente (Decision Tree e Random Forest aceitariam texto
# também, mas o XGBoost exige número — padronizar como número desde o
# início evita ter que converter depois, separadamente, para cada modelo).
X_treino, X_teste, y_treino, y_teste = train_test_split(
X, y, test_size=0.2, stratify=y, random_state=42
)
#%% PASSO 7: Treinar os 3 modelos representativos das duas semanas
modelo_arvore_podada = DecisionTreeClassifier(random_state=42, max_depth=4)
modelo_arvore_podada.fit(X_treino, y_treino)
modelo_rf = RandomForestClassifier(random_state=42, n_estimators=400, min_samples_leaf=5)
modelo_rf.fit(X_treino, y_treino)
modelo_xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=6,
random_state=42, eval_metric='logloss'
)
modelo_xgb.fit(X_treino, y_treino)
# Este dicionário guarda os 3 modelos já treinados, com um "apelido"
# (a chave do dicionário) para cada um — é o mesmo padrão que usamos na
# Semana 1 para gerar a tabela final e os relatórios em loop.
modelos = {
'Decision Tree (Semana 1)': modelo_arvore_podada,
'Random Forest (Semana 2)': modelo_rf,
'XGBoost (Semana 2)': modelo_xgb,
}
#%% PASSO 8: Montar a tabela de métricas para os 3 modelos de uma vez
resultados = []
# resultados começa como uma lista vazia. A cada volta do for abaixo,
# vamos adicionar um dicionário com as métricas daquele modelo — no
# final, transformamos essa lista de dicionários numa tabela (DataFrame).
for nome, modelo in modelos.items():
# a cada volta, "nome" e "modelo" mudam para o próximo par do
# dicionário (ver explicação detalhada disso nas mensagens anteriores
# desta conversa, se precisar relembrar como o for com .items() funciona).
    y_pred = modelo.predict(X_teste)
# a previsão final (0 ou 1) de cada cliente.
    y_prob = modelo.predict_proba(X_teste)[:, 1]
# predict_proba() (diferente de predict()) devolve, para cada
# cliente, DUAS probabilidades: a chance de ser classe 0 e a chance
# de ser classe 1, sempre somando 100%. O "[:, 1]" seleciona só a
# segunda coluna (a probabilidade de ser classe 1, ou seja, "Yes,
# vai cancelar") — é esse número, entre 0 e 1, que a curva ROC
# (Passo 9) e o roc_auc_score abaixo precisam, em vez da decisão
# final de 0/1.
    resultados.append({
            'Modelo': nome,
            'Accuracy': accuracy_score(y_teste, y_pred),
            'Precision': precision_score(y_teste, y_pred),
            'Recall': recall_score(y_teste, y_pred),
            'F1': f1_score(y_teste, y_pred),
            'ROC AUC': roc_auc_score(y_teste, y_prob),
# roc_auc_score resume a curva ROC inteira (que você já conhece
# do Dia 2) num único número entre 0 e 1: quanto mais perto de 1,
# melhor o modelo separa quem cancela de quem fica, considerando
# TODOS os limites de decisão possíveis, não só o padrão de 50%.
})
tabela_final = pd.DataFrame(resultados).set_index('Modelo')

# pd.DataFrame() transforma a lista de dicionários numa tabela de verdade.

# .set_index('Modelo') faz a coluna "Modelo" virar o rótulo de cada
# linha, em vez de uma coluna comum — deixa a tabela mais fácil de ler.
tabela_final = tabela_final.round(3)
# .round(3) arredonda todos os números para 3 casas decimais, só para
# a tabela impressa não ficar poluída com dízimas longas.
print("\nTabela comparativa final (as 2 semanas juntas):\n")
print(tabela_final)
tabela_final.to_csv('images/tabela_resultados_final.csv')
# salva a tabela como um arquivo de planilha simples (CSV), que você
# pode abrir depois no Excel ou colar direto no README do portfólio.
#%% PASSO 9: Curvas ROC dos 3 modelos no mesmo gráfico
fig, eixo = plt.subplots(figsize=(8, 6))
# plt.subplots() cria uma figura (fig) e um "eixo" de desenho (eixo) de
# uma vez só — é a forma recomendada quando você vai desenhar várias
# curvas SOBRE o mesmo gráfico, em vez de gráficos separados.
for nome, modelo in modelos.items():
    RocCurveDisplay.from_estimator(modelo, X_teste, y_teste, name=nome, ax=eixo)
# ax=eixo diz para essa curva ser desenhada no MESMO gráfico que as
# anteriores, em vez de abrir um gráfico novo a cada volta do for —
# é assim que as 3 curvas acabam sobrepostas na mesma imagem final.
    eixo.plot([0, 1], [0, 1], '--', color='grey', label='Classificador aleatório')
# a linha diagonal de referência, que representa um modelo que "chuta"
# sem aprender nada — qualquer curva acima dela está indo melhor que sorte.
    eixo.set_title('Comparação final — Decision Tree x Random Forest x XGBoost')
    eixo.legend(loc='lower right')
# loc='lower right' posiciona a caixinha de legenda no canto inferior
# direito do gráfico, onde normalmente não atrapalha as curvas (que
# tendem a subir para o canto superior esquerdo).
plt.tight_layout()
plt.savefig('images/roc_comparacao_final.png')
print("\nGráfico salvo em images/roc_comparacao_final.png")