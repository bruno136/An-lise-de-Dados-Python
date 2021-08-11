#Bibliotecas para auxiliar na manipulação dos dados

import numpy as np
import pandas as pd

#Lendo a base de dados
base_dados_integrada = pd.read_csv("BaseDadosIntegrada_2016_2017_2018_2019_2020.csv",encoding="utf-8")
print(base_dados_integrada);

# Seleção de variáveis preditoras (Feature Selection)
atributos = ['Nota_Final_AEDSI']

# Variável a ser prevista
atrib_prev = ['Nota_Final_AEDSII'];

# Criando objetos
X = base_dados_integrada[atributos].values
Y = base_dados_integrada[atrib_prev].values
print("\n",X);
print(Y);

#Função para converter as notas para valores booleanos
# que representa aprovado e reprovado na disciplina
def organiza_notas(val):
    if(val>= 60):
        return 1;
    else:
        return 0;

base_dados_integrada["Nota_Final_AEDSII"] = base_dados_integrada["Nota_Final_AEDSII"].map(organiza_notas)

# Seleção de variáveis preditoras (Feature Selection)
atributos = ['Nota_Final_AEDSI']

# Variável a ser prevista
atrib_prev = ['Nota_Final_AEDSII'];

# Criando objetos
X = base_dados_integrada[atributos].values
Y = base_dados_integrada[atrib_prev].values

print(X);
print(Y)
#Bibliotecas para aplicar a validação cruzada

#Lista para o cálculo das média das acurácias
lista = [];

from sklearn.model_selection import cross_val_score

#Algoritmos de Classificação

#Modelo Naive Bayes
from sklearn.naive_bayes import GaussianNB

# Criando o modelo preditivo
modelo_v1 = GaussianNB()

#Realizando a acurácia aplicando o modelo de validação cruzada
#cv também é definido como n_splits em que define em quantos conjutos é dividido em cada passada para realizar cross_validation
scores = cross_val_score(modelo_v1, X, Y.ravel(), cv=3)

#Acurácias nas 3 passadas
print("\n",scores);

#Acurácia média
print("Accuracy GaussianNB: %0.2f " % (scores.mean()))

print();
print(base_dados_integrada.corr());
print();

#Correlação individual dos objetos
print("\n",base_dados_integrada["Nota_Final_AEDSI"].corr(base_dados_integrada["Nota_Final_AEDSII"]))

#Modelo Random Forest
from sklearn.ensemble import RandomForestClassifier

modelo_v2 = RandomForestClassifier(random_state = 42)
#Realizando a acurácia aplicando o modelo de validação cruzada
#cv também é definido como n_splits em que define em quantos conjutos é dividido em cada passada para realizar cross_validation
scores = cross_val_score(modelo_v2, X, Y.ravel(), cv=3)

#Acurácias nas 3 passadas
print("\n",scores);

#Acurácia média
print("Accuracy RandomForestClassifier: %0.2f " % (scores.mean()))

#Modelo de rede neural MultiLayerPerception

from sklearn.neural_network import MLPClassifier

modelo_v3 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(3, 3), random_state=678900)

#Realizando a acurácia aplicando o modelo de validação cruzada
#cv também é definido como n_splits em que define em quantos conjutos é dividido em cada passada para realizar cross_validation
scores = cross_val_score(modelo_v3, X, Y.ravel(), cv=3)

#Acurácias nas 3 passadas
print("\n",scores);

#Acurácia média
print("Accuracy MLP: %0.2f " % (scores.mean()))


#Modelo DecisionTreeClassifier (Arvores de decisão)

from sklearn.tree import DecisionTreeClassifier

modelo_v4 = DecisionTreeClassifier(random_state=42);

#Realizando a acurácia aplicando o modelo de validação cruzada
#cv também é definido como n_splits em que define em quantos conjutos é dividido em cada passada para realizar cross_validation
scores = cross_val_score(modelo_v4, X, Y.ravel(), cv=3)

#Acurácias nas 3 passadas
print("\n",scores);

#Acurácia média
print("Accuracy DecisionTreeClassifier: %0.2f " % (scores.mean()))

#Modelo KNN(K-vizinhos mais próximos) Métodos de vizinhos mais próximos (KNeighborsClassifier)

from sklearn.neighbors import KNeighborsClassifier

modelo_v5 = KNeighborsClassifier(n_neighbors=14)

#Realizando a acurácia aplicando o modelo de validação cruzada
#cv também é definido como n_splits em que define em quantos conjutos é dividido em cada passada para realizar cross_validation
scores = cross_val_score(modelo_v5, X, Y.ravel(), cv=3)

#Acurácias nas 3 passadas
print("\n",scores);

#Acurácia média
print("Accuracy KNeighborsClassifier: %0.2f " % (scores.mean()))

#Modelo Regressão Logística
from  sklearn.linear_model import LogisticRegression

# Terceira versão do modelo usando Regressão Logística
modelo_v6 = LogisticRegression(random_state = 42)

#Realizando a acurácia aplicando o modelo de validação cruzada
#cv também é definido como n_splits em que define em quantos conjutos é dividido em cada passada para realizar cross_validation
scores = cross_val_score(modelo_v6, X, Y.ravel(), cv=3)

#Acurácias nas 3 passadas
print("\n",scores);

#Acurácia média
print("Accuracy LogisticRegression: %0.2f " % (scores.mean()))
