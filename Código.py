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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from math import nan
import shap
from sklearn.feature_selection import RFECV
from sklearn.model_selection import ShuffleSplit
from imblearn.over_sampling import SMOTE

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

for i in df_replace_nan.columns.values:
    df_replace_nan[i].isna().sum()/df_replace_nan.shape[0]


#Filtrando os NaN pela média de cada coluna
df_replace_nan = df_replace_nan.fillna(df_replace_nan.mean())

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
df_filter_volatilidade = df_filter_volatilidade.fillna(df_filter_volatilidade.mean())
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
                                                                 test_size=0.3)

#Balanceamento do dataset
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

#Seleção de features final utilizando o algoritmo Recursive Feature Selection (RFE)
#Amostrando e normalizando o conjunto de treino
scaler = StandardScaler()
X_sample = X_resampled.head(1000)
y_sample = y_resampled.head(1000)
X_sample[X_sample.columns] = scaler.fit_transform(X_sample[X_sample.columns])

estimator = LGBMClassifier(random_state=42)
cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
rfecv = RFECV(estimator, min_features_to_select=10, cv = cv)

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

sns.lineplot(results, x = range(64, 9, -1),y = 'mean_test_score')
plt.show()

#Obtendo os datasets com as variaveis resultantes
X_train_selected = rfecv.transform(X_resampled)
X_test_selected = rfecv.transform(X_validation)

estimator.fit(X_train_selected, y_resampled)
y_pred_selected = estimator.predict(X_test_selected)

#Calculando as métricas de desempenho
accuracy = accuracy_score(y_pred_selected, y_validation)
precision = precision_score(y_pred_selected, y_validation)
recall = recall_score(y_pred_selected, y_validation)
f1 = f1_score(y_pred_selected, y_validation)

# Print the R2 and RMSE
print('Acurácia:', accuracy, '; Precisão:', precision,
      'recall:', recall, "f1_score:", f1)