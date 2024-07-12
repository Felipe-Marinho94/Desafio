'''
Desafio Técnico - Vaga Cientista de Dados
Empresa - BIX Tecnologia
Nome: Felipe Pinto Marinho
Data:08/07/2024
'''

#--------------------------------------------------------------
#Carregando alguns pacotes relevantes
#--------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
from random import sample
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from math import nan
import shap
from sklearn.feature_selection import RFECV
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import label_binarize
from imblearn.under_sampling import  RandomUnderSampler
import optuna
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from optuna.samplers import TPESampler

#--------------------------------------------------------------
#Implementado algumas funções relevantes
#--------------------------------------------------------------
#Filtro de Volumetria com Threshold de 0.7
def filtro_volumetria(dataset):
    for i in dataset.columns.values:
        if dataset[i].isna().sum(axis = 0)/dataset.shape[0] > 0.7:
            dataset = dataset.drop([i], axis = 1)
    return(dataset)

#Filtro de Assimetria com Threshold de 2
def filtro_assimetria(dataset):
    for i in dataset.select_dtypes(include = ["int64", "float64"]).columns.values:
        if round(abs(dataset[i].skew()), 2) > 2:
            dataset = dataset.drop([i], axis = 1)
    return(dataset)

#Filtro de volatilidade com Threshold de 0.9
def filtro_volatilidade(dataset):
    for i in dataset.select_dtypes(include = ["int64", "float64"]).columns.values:
        if round(dataset[i].var(), 2) < 0.9:
            dataset = dataset.drop([i], axis = 1)
    return(dataset)

#Filtro de correlação com threshold de 0.8 e correlação de spearman
def remove_collinear_features(x, threshold):

    # Calcula a matriz de correlação de spearmen
    corr_matrix = x.corr(method = 'spearman')
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            if val >= threshold:
    
                drop_cols.append(col.values[0])

    
    drops = set(drop_cols)
    x = x.drop(columns=drops)
    print('Coluna removida {}'.format(drops))
    return x

def metricas(y_ground, y_hat):
    results = {'Accuracy': accuracy_score(y_ground, y_hat),
               'Precision':precision_score(y_ground, y_hat),
               'Recall': recall_score(y_ground, y_hat),
               'f1_score': f1_score(y_ground, y_hat),
               'balanced_accuracy': balanced_accuracy_score(y_ground, y_hat),
               'AUC': roc_auc_score(y_ground, y_hat)}
    return(results)

#Criando a métrica de custo
def custo(y_true, y_hat):
    '''
    Custo do falso positivo de 25 dólares
    Custo do falso negativo de 500 dólares
    '''
    tn, fp, fn, tp = confusion_matrix(y_validation, y_hat).ravel()
    return(fp * 25 + fn * 500)

#Rotulação alterando o ponto de corte
def to_labels(pos_probs, threshold):
 return (pos_probs >= threshold).astype('int')

#--------------------------------------------------------------
#Carregando base de dados
#--------------------------------------------------------------
df = pd.read_csv('air_system_previous_years.csv', sep = ',')
df.head()
df.tail()
df.describe()

#--------------------------------------------------------------
#Análise Exploratória de Dados
#--------------------------------------------------------------
#Converter o caracter 'na' em elemento Not a Number 'NaN'
df_replace_nan = df.replace('na', nan)

df_nulls_pos = df_replace_nan[df_replace_nan['class'] == 'pos'].isna().sum().sort_values(ascending = False)/df_replace_nan[df_replace_nan['class'] == 'pos'].shape[0]
df_nulls_neg = df_replace_nan[df_replace_nan['class'] == 'neg'].isna().sum().sort_values(ascending = False)/df_replace_nan[df_replace_nan['class'] == 'neg'].shape[0]

df_nulls_pos.head(10)
df_nulls_neg.head(10)

df_nulls_pos.loc[df_nulls_neg.head(10).index]
df_nulls_neg.loc[df_nulls_pos.head(10).index]

for i in ['br_000', 'bq_000', 'bp_000', 'bo_000']:
    for j in range(0, df_replace_nan.shape[0]):
        if df_replace_nan['class'][j] == 'neg':
            df_replace_nan[i][j] = 1
        else:
            df_replace_nan[i][j] = 0


new_features = df_replace_nan[['br_000', 'bq_000', 'bp_000', 'bo_000']]

new_features

#Filtrando os NaN pela média de cada coluna
df_replace_nan = df_replace_nan.fillna(0)

#Avaliação do balanceamento entre classes
balac = df_replace_nan.groupby(['class']).size().reset_index(name = 'counts')
balac

#Gráfico de barras para avaliação do desbalanceamento de classes
sns.set_theme(style="darkgrid")
plt.figure(figsize=(8, 8))
sns.barplot(x='class', y="counts",
             hue="class", data = balac,
             palette = 'mako')

plt.show()

#--------------------------------------------------------------
#Modelagem
#--------------------------------------------------------------
#Seleção de features (Análise Univariada)
#Filtro de Volumetria
df_filter_volumetria = filtro_volumetria(df_replace_nan)
df_replace_nan.shape
df_filter_volumetria.shape

#Filtro de Assimetria
df_filter_assimetria = filtro_assimetria(df_filter_volumetria)
df_filter_volumetria.shape
df_filter_assimetria.shape

#Filtro de Volatilidade
df_filter_volatilidade = filtro_volatilidade(df_filter_assimetria)
df_filter_assimetria.shape
df_filter_volatilidade.shape
df_filter_volatilidade.info()

#Seleção de features (Análise Bivariada)
#Matriz de correlação em forma de mapa de calor
col = df_filter_volatilidade.drop(['class'], axis=1).columns
df_filter_volatilidade[col] = df_filter_volumetria[col].apply(pd.to_numeric, errors='coerce')

#Filtrando os NaN pela média de cada coluna
df_filter_volatilidade = df_filter_volatilidade.fillna(0)
df_filter_volatilidade.head

#Removendo features com alta correlação
df_filter_correlation = remove_collinear_features(df_filter_volatilidade, 0.8)
df_filter_correlation.shape

plt.figure(figsize=(10,7))
mask = np.triu(np.ones_like(df_filter_correlation.iloc[:, 0:9].corr(numeric_only=True, method='spearman'), dtype=float))
sns.heatmap(df_filter_correlation.iloc[:, 0:9].corr(numeric_only=True), annot=True, mask=mask, vmin=-1, vmax=1)
plt.title('Matriz de correlação para os preditores numéricos')
plt.show()
df_filter_correlation.shape

#Tratamento de outliers utilizando o algoritmo 
# Ordering Points to Identify Cluster Structures (OPTICS)
#Instanciando o objeto da classe OPTICS
outliers_detector = OPTICS().fit(df_filter_correlation.drop(['class'], axis = 1))

#Obtendo as distâncias
scores = outliers_detector.core_distances_

#Estabelecendo um threshold
thresh = np.quantile(scores, 0.98)

#Detectando os outliers
index_outliers = np.where(scores >= thresh)
outliers = df_filter_correlation.iloc[index_outliers]
index_outliers = list(index_outliers)[0]

#Filtrando
index_selected = ~df_filter_correlation.index.isin(index_outliers)
df_filter_outlier = df_filter_correlation.loc[index_selected]
df_filter_outlier['class']
df_filter_outlier.shape

#Visualizando os outliers
df_filter_correlation.head
plt.scatter(df_filter_correlation.iloc[:, 2], df_filter_correlation.iloc[:, 3])
plt.scatter(outliers.iloc[:, 1],outliers.iloc[:, 2], color='r')
plt.legend(("normal", "anomal"), loc="best", fancybox=True, shadow=True)
plt.grid(True)
plt.show()

#Divisão Subtreino/Validação Estratificada baseada na resposta 'class'
X_train, X_validation, y_train, y_validation = train_test_split(df_filter_outlier.drop(['class'], axis = 1),
                                                                 df_filter_outlier['class'],
                                                                 stratify=df_filter_outlier['class'],
                                                                 test_size=0.3, random_state=42)
#Binarização dos rótulos
y_train = label_binarize(y_train, classes = ['neg', 'pos']).flatten()
y_validation = label_binarize(y_validation, classes = ['neg', 'pos']).flatten()

#Balanceamento do dataset
rus = RandomUnderSampler(random_state=42, sampling_strategy=0.25)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
X_resampled.shape
y_resampled

#Seleção de features final utilizando o algoritmo Recursive Feature Selection (RFE)
#Amostrando e normalizando o conjunto de treino
X_sample = X_resampled
y_sample = y_resampled

estimator = LGBMClassifier()
cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
rfecv = RFECV(estimator, min_features_to_select=1, cv = cv)

#Fittando nos dados
rfecv.fit(X_sample, y_sample)

#Indices dos features selecionados
feature_index = rfecv.get_support(indices=True)

#Obtendo uma máscara com os features selcionados
feature_mask = rfecv.support_

#Nomes dos features selecionados
feature_names = rfecv.get_feature_names_out()

#Número de features selecionados
feature_number = rfecv.n_features_

#Resultados
results = pd.DataFrame(rfecv.cv_results_)

#Obtendo RFECV score
rfecv_score = rfecv.score(X_sample, y_sample)

# Print número de features
print('Original feature number:', len(X_sample.columns))
print('Optimal feature number:', feature_number)
print('Selected features:', feature_names)
print('Score:', rfecv_score)

sns.lineplot(results, x = range(75, 0, -1),y = 'mean_test_score')
plt.show()
results

#Obtendo os datasets com as variaveis resultantes
X_train_selected = rfecv.transform(X_resampled)
X_test_selected = rfecv.transform(X_validation)
X_train_selected.shape
X_test_selected.shape
y_resampled.shape
y_validation
estimator.fit(X_train_selected, y_resampled)
y_pred_selected = estimator.predict(X_test_selected)

#Calculando as métricas de desempenho
accuracy = accuracy_score(y_validation, y_pred_selected)
precision = precision_score(y_validation, y_pred_selected, average="binary" ,pos_label=1)
recall = recall_score(y_validation, y_pred_selected, average="binary" ,pos_label=1)
f1 = f1_score(y_validation, y_pred_selected, average="binary" ,pos_label=1)
accuracy_balanced = balanced_accuracy_score(y_validation, y_pred_selected)

for i in y_pred_selected:
    print(i)


metricas(y_validation, y_pred_selected)

#Modelos
models = {
    'REG_LOG': LogisticRegression(max_iter=10000),
    'KNN': KNeighborsClassifier(),
    'MLP': MLPClassifier(),
    'CART': DecisionTreeClassifier(),
    'LightGBM': LGBMClassifier(),
    'XGBoost': XGBClassifier()
}

#Treinando e obtendo previsões
train_hat, test_hat = {}, {}
for k in models:
    models[k].fit(X_train_selected, y_resampled)
    train_hat[k] = models[k].predict(X_train_selected)
    test_hat[k] = models[k].predict(X_test_selected)

ts_hat_df = pd.DataFrame(test_hat)
tr_hat_df = pd.DataFrame(train_hat)

#Avaliação
metricas(tr_hat_df['KNN'], y_resampled)
metricas(ts_hat_df['REG_LOG'], y_validation)
metricas(ts_hat_df['MLP'], y_validation)
metricas(ts_hat_df['XGBoost'], y_validation)
metricas(ts_hat_df['LightGBM'], y_validation)


#Otimizando o KNN usando o optuna
def objective(trial):
    #Normalização
    scalers = trial.suggest_categorical("scalers", ['minmax', 'standard', 'robust'])

    if scalers == "minmax":
        scaler = MinMaxScaler()
    elif scalers == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
                
    #Vizinhos, Ponderação e Distâncias
    n_neighbors = trial.suggest_int("n_neighbors", 1, 30)
    weights = trial.suggest_categorical("weights", ['uniform', 'distance'])
    metric = trial.suggest_categorical("metric", ['euclidean', 'manhattan', 'minkowski'])
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
        
    # -- Make a pipeline
    pipeline = make_pipeline(scaler, knn)

    kfold = StratifiedKFold(n_splits=3)
    score = cross_val_score(pipeline, X_test_selected, y_validation, scoring='accuracy', cv=kfold)
    score = score.mean()
    return score

sampler = TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=20)

optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)

study.best_params


#Otimização do modelo LightGBM usando optuna
study = optuna.create_study(direction="maximize")

def objective(trial):
  #Hiperparâmetros e seus espaços de busca
  learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1)
  num_leaves = trial.suggest_int("num_leaves", 2, 256)
  max_depth = trial.suggest_int("max_depth", -1, 50)
  min_child_samples = trial.suggest_int("min_child_samples", 5, 100)
  subsample = trial.suggest_float("subsample", 0.5, 1.0)
  colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
  n_estimators = trial.suggest_int("n_estimators", 100, 1000)
  
  #Criando e treinando o modelo
  model = LGBMClassifier(
  learning_rate=learning_rate,
  num_leaves=num_leaves,
  max_depth=max_depth,
  min_child_samples=min_child_samples,
  subsample=subsample,
  colsample_bytree=colsample_bytree,
  n_estimators=n_estimators,
  random_state=42
  )
  model.fit(X_train_selected, y_resampled)
  
  #Avaliação do modelo e métrica
  y_pred = model.predict(X_test_selected)
  f1 = f1_score(y_validation, y_pred)
  return f1

#Otimizando
study.optimize(objective, n_trials=20)
print("Best trial:")
print(" Value: {}".format(study.best_trial.value))
print(" Params: {}".format(study.best_trial.params))



estimator = LGBMClassifier(
  learning_rate= 0.010249181370892781,
  num_leaves=256,
  max_depth=13,
  min_child_samples=27,
  subsample=0.9745196507456092,
  colsample_bytree=0.6386493013874187,
  n_estimators=108,
  random_state=42
  )

estimator = KNeighborsClassifier(n_neighbors=6,
                                 weights='distance',
                                 metric='minkowski')

estimator.fit(X_train_selected, y_resampled)
y_hat = estimator.predict(X_test_selected)
metricas(y_hat, y_validation)

#Otimizando o ponto de corte usando a métrica custo
#Estimando as probabiblidades
y_hat_probs = estimator.predict_proba(X_test_selected)
y_hat_probs[:,1]

#Definindo pontos de cortes arbitrários
thresholds = np.arange(0, 1, 0.001)

#Avaliando o custo para cada ponto de corte
scores = [custo(y_validation, to_labels(y_hat_probs[:, 1], t)) for t in thresholds]

#Obtendo o melhor ponto de corte
ix = np.argmin(scores)
print('Threshold=%.3f, Custo=%.5f' % (thresholds[ix], scores[ix]))
optimum_Threshold = thresholds[ix]

#Avaliando no teste
df_test = pd.read_csv('air_system_present_year.csv', sep=',')
df_test.head

#Subsituindo os na por nan
df_test_replace_nan = df_test.replace('na', nan)
df_test_replace_nan = df_test_replace_nan.fillna(0)
col = df_test_replace_nan.drop(['class'], axis=1).columns
df_test_replace_nan[col] = df_test_replace_nan[col].apply(pd.to_numeric, errors='coerce')
X_test = df_test_replace_nan[feature_names]
y_test = label_binarize(df_test_replace_nan['class'], classes = ['neg', 'pos']).flatten()

y_hat_test_probs = estimator.predict_proba(X_test)
y_hat_test = to_labels(y_hat_test_probs[:, 1], optimum_Threshold)
metricas(y_test, y_hat_test)