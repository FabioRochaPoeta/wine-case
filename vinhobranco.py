# %% [markdown]
# # Vamos analisar vinhos brancos!
# 
# Usaremos um dataset composto por propriedades físico-químicas de vinhos brancos. Temos 6497 amostras e um total de 11 variáveis independentes mais o tipo de vinho (type), descritas abaixo:
# 
#  - `fixed acidity`: a maioria dos ácidos envolvidos com vinho (não evaporam prontamente)
#  - `volatile acidity`: a quantidade de ácido acético no vinho, que em níveis muito altos pode levar a um gosto desagradável de vinagre
#  - `citric acid`: encontrado em pequenas quantidades, o ácido cítrico pode adicionar "leveza" e sabor aos vinhos
#  - `residual sugar`: a quantidade de açúcar restante após a fermentação é interrompida, é raro encontrar vinhos com menos de 1 grama / litro e vinhos com mais de 45 gramas / litro são considerados doces
#  - `chlorides`: a quantidade de sal no vinho
# free sulfur dioxide: a forma livre de SO2 existe em equilíbrio entre o SO2 molecular (como gás dissolvido) e o íon bissulfito; impede o crescimento microbiano e a oxidação do vinho
#  - `total sulfur dioxide`: Quantidade de formas livres e encadernadas de S02; em baixas concentrações, o SO2 é quase indetectável no vinho, mas nas concentrações de SO2 acima de 50 ppm, o SO2 se torna evidente no nariz e no sabor do vinho.
#  - `density`: a densidade do vinho é próxima a da água, dependendo do percentual de álcool e teor de açúcar
#  - `pH`: descreve se o vinho é ácido ou básico numa escala de 0 (muito ácido) a 14 (muito básico); a maioria dos vinhos está entre 3-4 na escala de pH
#  - `sulphates`: um aditivo de vinho que pode contribuir para os níveis de gás de dióxido de enxofre (S02), que age como um antimicrobiano e antioxidante
#  - `alcohol`: o percentual de álcool no vinho
# 
# 
# Existe ainda uma variável chamada `quality`. Essa variável é uma nota de qualidade do vinho que varia de 0 a 10.

# %% [markdown]
# # Trabalho:
# 
# Kaggle
# 
# Link do certificado: https://drive.google.com/file/d/1QYMGnxyC0iaKSURAq55OwIg59pef9hv4/view?usp=share_link
# 
# ![](https://drive.google.com/file/d/1QYMGnxyC0iaKSURAq55OwIg59pef9hv4/view?usp=share_link)

# %% [markdown]
# Faça o download da base - esta é uma base real, apresentada no artigo:
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

# %%
# Base baixada, criando dataframe e importando tudo:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    RocCurveDisplay
)

sns.set_style("ticks")
sns.set_context("paper")

wine = pd.read_csv('winequalityN.csv', sep=",")

wine.head()

# %% [markdown]
# Ela possui uma variável denominada "quality", uma nota de 0 a 10 que denota a qualidade do vinho. Crie uma nova variável, chamada "opinion" que será uma variável categórica igual à 0, quando quality for menor ou igual à 5. O valor será 1, caso contrário. Desconsidere a variável quality para o restante da análise.

# %%
# criando coluna opinion 

wine['opinion'] = (wine['quality'] > 5).astype(int)
wine

# %%
# analisando dados: numeros de red e de white

wine['type'].value_counts()

# %%
# separando em novo dataframe apenas vinhso brancos e dropando a coluna quality

color = 'white'
df_white = wine.drop('quality', axis=1).query('type == @color').copy()

# %%
# mostrando novo dataframe só com vinhos brancos

df_white

# %% [markdown]
# Descreva as variáveis presentes na base. Quais são as variáveis? Quais são os tipos de variáveis (discreta, categórica, contínua)? Quais são as médias e desvios padrões?

# %%
# médias e desvios padrões:

df_white.describe()

# %%
# tipos:

df_white.dtypes

# %%
# detalhando tipos

print('object  = não-numérico   = variável categórica (qualitativa nominal, no caso)')
print('float64 = número real    = variável contínua')
print('int64   = número inteiro = variável discreta')


# %% [markdown]
# Com a base escolhida:
# 
# Descreva as etapas necessárias para criar um modelo de classificação eficiente.
# 
# - 1 - Limpar NaNs e outliers (usando melhor método - moda, média, regressão linear etc.)
# - 2 - Escolha do modelo
# - 3 - Separar conjunto de treino e teste
# - 4 - Treinar modelo
# - 5 - Rodar modelo com variáveis de teste para comparar o yhat/ypred (previsto no teste) com os targets do y de treino (geralmente y_test)
# - 6 - Utilizar f1-score, acurácia, precisão e recall para analisar a eficiência.

# %% [markdown]
# Treine um modelo de regressão logística usando um modelo de validação cruzada estratificada com k-folds (k=10) para realizar a classificação. Calcule para a base de teste:
# i. a média e desvio da acurácia dos modelos obtidos;
# ii. a média e desvio da precisão dos modelos obtidos;
# iii. a média e desvio da recall dos modelos obtidos;
# iv. a média e desvio do f1-score dos modelos obtidos.

# %%
# Checando DF

df_white.head()




# %%
df_white.shape

# %%
# Checando NaNs

df_white.info()


# %%
# NaNs 

print(df_white.isnull().sum())

# %%
# Retirando NaNs

df_white.dropna(inplace=True)

# %%
# Checando Nans

print(df_white.isnull().sum())

# %%
# Resetando índice após apagar as linhas com NaN

df_white.reset_index(inplace=True)

# %%
df_white

# %%
# Apagando índice antigo

df_white = df_white.drop('index', axis=1)

# %%
df_white

# %%
# Checando Outliers - aparentemente não tem (ignorando opinion, que é o target) - rever se sobrar tempo um método mais científico

df_white.drop('opinion', axis=1).hist(figsize=[16,16]);

# %%
# Checando Outliers com Boxplot (mais preciso)
 
df_white.drop('opinion', axis=1).boxplot(figsize=[12,8]);

# %%
# Dropando free sulfer dioxide >200 - Possível Outlier mais claro


outlier = df_white[df_white['free sulfur dioxide'] > 200 ].index

df_white = df_white.drop(index = outlier)

# %%
# Iniciando regressão logística

vars = [
   'fixed acidity',
   'volatile acidity',
   'citric acid',
   'residual sugar',
   'chlorides',
   'free sulfur dioxide',
   'total sulfur dioxide',
   'density',
   'pH',
   'sulphates',
   'alcohol'
]

# %%
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
sns.heatmap(df_white[vars].corr(), vmax=1, vmin=-1, annot=True, ax=ax, cmap="coolwarm");

# %%
X_train, X_test, y_train, y_test = train_test_split(df_white[vars],
                                                    df_white['opinion'],
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=df_white['opinion'])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
logreg = LogisticRegression(max_iter=10000)


logreg.fit(X_train_scaled, y_train)

# %%
y_hat = logreg.predict_proba(X_train_scaled)
print(y_hat.shape)

# %%
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
sns.histplot(y_hat[y_train.values == 1, 1], label="Good", ax=ax)
ax.set_xlim([0, 1])
sns.histplot(y_hat[y_train == 0, 1], label="Bad", ax=ax)
ax.legend();
ax.axvline(0.5, color="red", ls=":", lw=3);

# %%
y_pred = logreg.predict(X_train_scaled)

cm = confusion_matrix(y_train, y_pred)

#cm = np.array([[434, 161], [1168, 516]])

ax = sns.heatmap(cm, cmap="BuGn", annot=True, fmt='g')
ax.set_xlabel("Predição")
ax.set_ylabel("Realidade")
ax.set_title("Matriz de Confusão")

ax.set_xticklabels(["Ruim (0)", "Bom (1)"]);
ax.set_yticklabels(["Ruim (0)", "Bom (1)"]);

# %%
y_pred

# %%
precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])

print(f"A precisão é {100*  precision:.2f} %")

accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)

print(f"A acurácia é {100 * accuracy:.2f} %")

# Sensibilidade ou taxa de verdadeiro positivo
sensibility = (cm[1, 1] / (cm[1, 1] + cm[1, 0]))
print(f"A sensibilidade é {100 *  sensibility:.2f} %")

specificity = (cm[0,0] / (cm[0, 0] + cm[0 ,1])) 

print(f"A especificidade é {100 * specificity:.2f} %")

# (1 - specificity) ou taxa de falsos positivos 

F1_score = 2 *(sensibility * precision) / (sensibility + precision)

print(f"F1 Score =  {F1_score:.2f}")

# %%
def plot_distributions(model, X, y, ax=None):
    y_hat = model.predict_proba(X)
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.histplot(y_hat[y.values == 1, 1], label="Good", ax=ax)
    ax.set_xlim([0, 1])
    sns.histplot(y_hat[y == 0, 1], label="Bad", ax=ax)
    ax.legend();
    return ax

ax = plot_distributions(logreg, X_train_scaled, y_train)


ax.axvline(0.5, lw=3, color='red', ls=":");
ax.axvline(0.75, lw=2, color='gray', ls="--");

# %%
print(f"A acurácia é {100 * accuracy_score(y_train, y_pred):.2f} %")
print(f"A sensibilidade é {100 *  recall_score(y_train, y_pred):.2f} %")
print(f"A precisão é {100*  precision_score(y_train, y_pred):.2f} %")

# %%
print(classification_report(y_train, y_pred))

# %%
thresholds = np.linspace(0, 1, 101)
y_hat = logreg.predict_proba(X_train)


for threshold in thresholds:
    predictions = []
    if (y_hat > threshold).any():
        y_pred = 1.
    else:
        y_pred = 0.
    predictions.append(y_pred)

# %%
y_hat = logreg.predict_proba(X_train_scaled)
thresholds = np.linspace(0, 1, 101)

def specificity_score(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    specificity = (cm[0,0] / (cm[0, 0] + cm[0 ,1])) 
    return specificity

def predict(model, X, threshold, pos_label=1):
    y_hat = model.predict_proba(X)
    y_pred = []
    for prob_tuple in y_hat:
        prob = prob_tuple[pos_label]
        if (prob > threshold).any():
            y_pred.append(1.)
        else:
            y_pred.append(0.)
    return np.array(y_pred)

#def predict(model, X, threshold, pos_label=1):
#    y_hat = model.predict_proba(X)
#    y_pred = (y_hat[:, pos_label] > threshold)
#    return y_pred.astype(float)
# Probabilidade de ter um vinho bom

recall = []
precision = []
specificity = []
f1 = []
for threshold in thresholds:
    y_pred_thr = predict(logreg, X_train_scaled, threshold)
    recall.append(recall_score(y_train, y_pred_thr))
    precision.append(precision_score(y_train, y_pred_thr))
    specificity.append(specificity_score(y_train, y_pred_thr))
    f1.append(f1_score(y_train, y_pred_thr))

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(thresholds, recall, color="orange", label="recall")
ax.plot(thresholds, precision, color="navy", label="precision")
ax.plot(thresholds, f1, color="olive", label="f1")

f1_max = max(f1)
thr_arg_max = np.argmax(f1)
thr_max = thresholds[thr_arg_max]

ax.axvline(thr_max, color="red", ls=":")
ax.axvline(0.5, color="gray",lw=0.5, ls="--")
ax.axhline(f1_max, color="red", ls=":")

ax.legend()
ax.set_ylim([0.5, 1])
print(f"f1 máximo: {f1_max:.2f} - ponto de operação: {thr_max:.2f}")
print(f"Recall: {recall[thr_arg_max]:.2f} - Precision: {precision[thr_arg_max]:.2f}")
sns.despine(offset=10)

# %%
# True positive rate ou taxa de verdadeiro positivo
tpr = recall
# False positive rate ou taxa de falso positivos
fpr = [(1 - s) for s in specificity]

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(fpr, tpr, color="navy")
ax.set_ylabel("Taxa de verdadeiro positivo")
ax.set_xlabel("Taxa de falso positivo")

ax.axvline(fpr[thr_arg_max], color="red", ls=":")
ax.axhline(tpr[thr_arg_max], color="red", ls=":")
ax.plot(thresholds, thresholds, color= "gray", ls=":", lw=0.5)
ax.set_ylim([0., 1.])
ax.set_xlim([0., 1.])

# %%
auc_score = auc(fpr, tpr)
print(f"Area Under Curve (AUC): {auc_score:.2f}")

# %%
def get_f1_score_list(model, X, y, thresholds):
    list_of_f1 = []
    for threshold in thresholds:
        y_pred = predict(model, X, threshold)
        f1 = f1_score(y, y_pred)
        list_of_f1.append(f1)
    return list_of_f1

def get_max_f1_score(model, X, y, thresholds):
    list_of_f1 = get_f1_score_list(model, X, y, thresholds)
    f1_max = max(list_of_f1)
    f1_arg_max = np.argmax(list_of_f1)
    threshold_max = thresholds[f1_arg_max]
    return f1_max, threshold_max, f1_arg_max


fpr, tpr, thresholds = roc_curve(y_train, y_hat[:, 1], pos_label=1)
auc_score = auc(fpr, tpr)
f1_max, threshold_max, f1_arg_max =  get_max_f1_score(logreg,
                                                      X_train_scaled,
                                                      y_train,
                                                      thresholds)


print(f"Area Under Curve (AUC): {auc_score:.2f}")
print(f"Maximum F1 : {f1_max:.2f} at {threshold_max:.3f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

threshold_random = 500

# DISTRIBUTION
plot_distributions(logreg, X_train_scaled, y_train, ax=axes[0])
axes[0].axvline(threshold_max, color="red", ls=":")
axes[0].axvline(thresholds[threshold_random], color="gray", ls=":")

# ROC CURVE
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_score).plot(ax=axes[1],color="green")
axes[1].axvline(fpr[f1_arg_max], color="red", ls=":")
axes[1].axhline(tpr[f1_arg_max], color="red", ls=":")
axes[1].annotate(f"Maximum F1 : {f1_max:.2f}", (fpr[f1_arg_max], tpr[f1_arg_max] - 0.05))


f1_list = get_f1_score_list(logreg, X_train_scaled, y_train, thresholds)


axes[1].axvline(fpr[threshold_random], color="gray", ls=":")
axes[1].axhline(tpr[threshold_random], color="gray", ls=":")

axes[1].annotate(f"F1 : {f1_list[threshold_random]:.2f}",
                        (fpr[threshold_random],
                         tpr[threshold_random] - 0.05))


# ROC CURVE Test
y_hat_test = logreg.predict_proba(X_test_scaled)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_hat_test[:, 1], pos_label=1)
auc_score_test = auc(fpr_test, tpr_test)

RocCurveDisplay(fpr=fpr_test, tpr=tpr_test, roc_auc=auc_score_test).plot(ax=axes[1],
                                                                         label=f"AUC (Test) = {auc_score_test:.2}",
                                                                         color="magenta")

# %% [markdown]
# Treine um modelo de árvores de decisão usando um modelo de validação cruzada estratificada com k-folds (k=10) para realizar a classificação. Calcule para a base de teste:
# i. a média e desvio da acurácia dos modelos obtidos;
# ii. a média e desvio da precisão dos modelos obtidos;
# iii. a média e desvio da recall dos modelos obtidos;
# iv. a média e desvio do f1-score dos modelos obtidos.

# %% [markdown]
# # Treinamento com validação cruzada
# 
# ![](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png)

# %%
# Import necessary packages
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import deepcopy as cp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.tree import (
    DecisionTreeClassifier, 
    plot_tree
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    RocCurveDisplay
)

sns.set_style("ticks")
sns.set_context("paper")

random_state = 42

X = df_white[vars]
y = df_white['opinion']

X_train_cv, X_test, y_train_cv, y_test = train_test_split(X.values,
                                                          y.values,
                                                          test_size=0.2, # 20 % da base
                                                          random_state=42,
                                                          stratify=y)

def train(X, y, model_klass, model_kwargs = {}):
    cv = StratifiedKFold(n_splits=10) # Esse n_splits é o K, do K folds, que no exercício é pedido 10
    f1_score_val_list = []
    f1_score_train_list = []
    model_list =[]
    scaler_list = []
    # Validação cruzada só em Training Data
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        X_val = X[val_idx, :]
        y_val = y[val_idx]

        # Escala
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        scaler_list.append(scaler)

        # Treino
        model = model_klass(**model_kwargs)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_train_scaled)

        y_pred_val = model.predict(X_val_scaled)
        print(f"========================= FOLD {fold} ==========================")
        print(f"Meu resultado para treino de F1-Score é {f1_score(y_train, y_pred):.2}")
        print(f"Meu resultado para validação de F1-Score é {f1_score(y_val, y_pred_val):.2}") 
        f1_score_val_list.append(f1_score(y_val, y_pred_val))
        f1_score_train_list.append(f1_score(y_train, y_pred))
        model_list.append(model)
    print()
    print()
    mean_val = np.mean(f1_score_val_list)
    std_val = np.std(f1_score_val_list)
    print(f"Meu resultado de F1-Score Médio de treino é {np.mean(f1_score_train_list): .2} (média) +- {np.std(f1_score_train_list): .2} (desvio padrão)")
    print(f"Meu resultado de F1-Score Médio de validação é {mean_val: .2} (média) +- {std_val: .2} (desvio padrão)")
    print()

    best_model_idx = np.argmax(f1_score_val_list)
    print(f"Meu melhor fold é: {best_model_idx} ")
    best_model = model_list[best_model_idx]

    # Fazer a inferência em Test Data
    best_scaler = scaler_list[best_model_idx]
    X_test_scaled = best_scaler.transform(X_test)
    y_pred_test = model.predict(X_test_scaled)

    print()
    print()
    print(f"Meu resultado de F1-Score para o conjunto de teste é: {f1_score(y_test, y_pred_test):.2} ")
    print(f"Melhor modelo, média, desvio padrão: {best_model, mean_val, std_val}")
    return best_model, mean_val, std_val

# %%
train(X_train_cv, y_train_cv, LogisticRegression)

# %%
# Fazendo árvore de decisão

tree_model, _, _  = train(X_train_cv, y_train_cv, DecisionTreeClassifier)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(60, 60))
plot_tree(tree_model, filled=True);

# %% [markdown]
# Treine um modelo de SVM usando um modelo de validação cruzada estratificada com k-folds (k=10) para realizar a classificação. Calcule para a base de teste:
# i. a média e desvio da acurácia dos modelos obtidos;
# ii. a média e desvio da precisão dos modelos obtidos;
# iii. a média e desvio da recall dos modelos obtidos;
# iv. a média e desvio do f1-score dos modelos obtidos.

# %%
# Iniciando SVM

from sklearn.svm import SVC
tree_model, _, _ = train(X_train_cv, y_train_cv, SVC, model_kwargs={'gamma': 'auto', 'C': 1, 'kernel': 'rbf'})

# %%
degrees = [3, 5, 10] 

results = []
for degree in degrees:
    tree_model_poly, mean_val, std_val = train(X_train_cv, y_train_cv, SVC, model_kwargs={'gamma': 'auto',
                                                                                          'C': 1,
                                                                                          'degree': degree,
                                                                                          'kernel': 'poly'})
    results.append(mean_val)

# %%
results

# %%
config = [
    (SVC, {'kernel': 'rbf'}),
    (SVC, {'kernel': 'rbf', 'gamma': 2}),
    (SVC, {'degree': 3, 'kernel': 'poly'} ),
    (SVC, {'degree': 5, 'kernel': 'poly'} ),
    (SVC, {'degree': 10, 'kernel': 'poly'} ),
    (LogisticRegression, {}),
    (DecisionTreeClassifier, {'min_samples_leaf': 50}),
]

results = []
for model_class, setting in config:
    print(model_class.__name__)
    best_model, mean_val, std_val = train(X_train_cv, y_train_cv, model_class, setting)
    results.append(mean_val)

# %%
results

# %% [markdown]
# Em relação à questão anterior, qual o modelo deveria ser escolhido para uma eventual operação. Responda essa questão mostrando a comparação de todos os modelos, usando um gráfico mostrando a curva ROC média para cada um dos gráficos e justifique.
# 
# 

# %%
# 0,83 - melhor resultado foi o SVM



# %%


# %% [markdown]
# Com a escolha do melhor modelo, use os dados de vinho tinto, presentes na base original e faça a inferência (não é para treinar novamente!!!) para saber quantos vinhos são bons ou ruins. Utilize o mesmo critério utilizado com os vinhos brancos, para comparar o desempenho do modelo. Ele funciona da mesma forma para essa nova base? Justifique.
# 
# Disponibilize os códigos usados para responder da questão 2 a 6 em uma conta github e indique o link para o repositório.
# 
# https://github.com/FabioRochaPoeta/wine-case/blob/main/analisando-db-vinho-branco.ipynb
# 
# Assim que terminar, salve o seu arquivo PDF e poste no Moodle. Utilize o seu nome para nomear o arquivo, identificando também a disciplina no seguinte formato: “nomedoaluno_nomedadisciplina_pd.PDF”.
# 
# OK


