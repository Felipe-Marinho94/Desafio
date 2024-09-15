## Neste Desafio Técnico, realizo a atividade em duas etapas:
* A primeira, faço todo o pré-processamento do *dataset* para realizar a identifição de sentimentos via algoritmos de aprendizagem de máquina clássicos (KNN, CART, *Random Forests* e *Support Vector Classifier* - SVC), utilizando apenas as colunas numéricas da base de modelagem (Arquivo *Sentiment_Analysis.ipynb*);
* Na segunda, importo o *dataset* somente com as colunas textuais *Review* e *Review_Title*, bem como, a variável alvo *sentiment* derivada da coluna *Overral_Rating* da base de modelagem original. Este novo *dataset* é importado para um *Google Colab* com o intuito de ser entrada para a realização de um *Fine-Tuning* de um modelo BERT pré-treinado, onde o uso do ambiente *Colab* é justificado pelo uso de recursos de GPU gratuito (Arquivo *BERT.ipynb*);

### Conclusões
* Os resultados são bem similares, ao utilizar a metodologia clássica, verificou-se que o modelo com melhor desempenho preditivo para a tarefa foi o SVC, que após um *tuning* de seus hiperparâmetros resultou em uma acurácia média de 0,78 (78%);
* O resultado pela utilização do modelo BERT foi de uma acurácia média de 0,73 (73%), utilizando apenas 10 épocas para treinamento do classificador, sendo este valor justificado pelas limitações de *Hardware*.
