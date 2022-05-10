#!/usr/bin/env python
# coding: utf-8

# # Notre démarche pour le projet : Credit Cards Fraud Detection 
Projet réalisé par: Asmae MELLIANI 
# # Phase 1: Pré-traitement des données 

# Importation des Bibliothèques nécessaires 

# In[47]:


import sys
print('Python: {}'.format(sys.version))
# scipy https://www.scipy.org/
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy https://numpy.org/
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib https://matplotlib.org/
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas https://pandas.pydata.org/
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn https://scikit-learn.org/stable/
import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# In[48]:


#load libraries
from pandas import read_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
import seaborn as sns
import matplotlib.patches as mpatches
import time 
import warnings
colors = ["#0101DF", "#DF0101"]
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
warnings.simplefilter('ignore')


RANDOM_SEED = 42


# Chargement des données 

# Cette étape consiste à charger les données telles qu'elles sont dans la dataset choisie

# In[49]:


# Load dataset
dataset = pd.read_csv('creditcard.csv')
dt = dataset.copy() # To keep the data as backup
print(dataset.columns)


# In[50]:


dt.head()


# In[51]:


dt.tail()


# In[52]:


dt.shape


#  Dans notre dataset réelle, nous avons : 
# 
#  284807 rows  et  31 columns 

# Pour la description de notre dataset : 
# 

# In[53]:


dt.describe()


# Exploration des données: 

# In[12]:


dt.info()


# Nettoyage des données:

# In[54]:


dt.isnull().sum().max()


# In[55]:


dt.isnull().sum()


# In[56]:


dt.isnull().values.any()


# In[57]:


dt['Class'].value_counts()


# On peut intrepréter ces résultats comme suit : 

# 1 signifie Fraud ==> 492 frauds 

# 0 signifie Legit ==> 284315 legits 

# In[58]:


LEGIT = dt[dt.Class == 0]
FRAUD = dt[dt.Class == 1]
print(LEGIT.shape)
print(FRAUD.shape)


# In[59]:


dt.groupby('Class').mean()


# Pour s'assurer de notre démarche, on pourra visualiser les résultats obtenus dans la section suivante qui sera consacrée à la visualisation.

# # Phase 2: Visualisation des données 

# En effet, cette étape permet la visualisation sous formats des figures et tableaux ,la distribution des
# différentes données de notre dataset (creditcard.csv) en fonctions de différents facteurs.
# 
Vérification de la fréquence des fraudes au départ: 
# In[119]:



plt.figure(figsize=(5,20))
count_classes = pd.value_counts(dt['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction Class Distribution")
LABELS = ['Legit','Fraud']
plt.xticks(range(2), LABELS)
plt.xlabel("Class of transaction: Class")


# EN EFFET, ce résultat est tout à fait normal, vu qu'on a 284315 cas legitimes et 492 cas de fraudes, ce qui fait qu'on ne peut pas visualiser les données des fraudes avec  des dimensions réelles assez grandes qui ne sont pas encore transformées.

# In[61]:


plt.figure(figsize=(10, 5))
sns.distplot(dt.Class)
plt.title("Transaction  Distribution")

Cette courbe illustre la desrtribution des cas légitimes et cas fraud : on remarque que le cas légitime (Normal) est plus dominant.
# 

# In[62]:


plt.figure(figsize=(10, 5))
sns.distplot(dt.Amount)
plt.title("Transaction Amount Distribution")

On remarque la densité des destributions de la classe Amount sur les 800-1000 premières données est bien élevée, mais depuis ce chiffre elle a chuté rapidement jusqu'elle est devenue nulle sur tout le reste de l'ensemble.

# 

# In[63]:


plt.figure(figsize=(10, 5))
sns.distplot(dt.Time)
plt.title("Transaction Time Distribution")

On remarque que la densité des destributions de la classe Time (en secondes) est variable sur l'ensemble de notre dataset de traitement, donc elle n'est pas stable. Du coup c'est difficile de prédire le temps de détection des fraudes.
# 
De meme, on peut générer cette fois-ci une matrice de corrélation pour toute les dimensions de notre dataset : creditcard.csv
# In[25]:


#get correlations of each features in dataset
corrmat = dt.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(15,15))
plt.title("Matrice de corrélation de fraud")
#plot heat map
g=sns.heatmap(dt[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# Les varibales ne sont pas bien corrélées.

# Maintenant, on présente sous forme d'histogramme l'avancement de toutes les dimensions de notre dataset: 

# In[65]:


#visual representation of the data using histograms 
dt.hist(figsize = (15, 15))
plt.show()


# La visualisation ci dessus nous permet de voir les transactions de chacune des 
# variables sous format des histogrammes.

# NB : Le travail sur des données de grandes dimensions ne donnent pas toujours des résultats satisfaisantes, raison pour laquelle nous aurons besoin d'une solution optimale pour la réduction des données.
# Ceci sera réalisé dans la prochaine étape. 

# # Phase 3: Extraction des caractéristiques
# 

# # Réduction de jeu de données

# L'extraction des caractéristiques consiste à reduire le jeu de données afin d'établir des modèles optimaux et de fournir  des résultats satisfaisants. 
# Nous allons essayer de présenter deux méthodes vues dans le cours du Machine Learning : PCA et chi-square 

# # Algorithme PCA : 

# L'avantage de cette dataset sur laquelle on a travaillé c'est qu'elle est dejà standarisée.
# Les caractéristiques V1, V2, … V28 sont dejà obtenues avec PCA, les seules caractéristiques qui n'ont pas été transformées avec PCA sont 'Time' et 'Amount'.
# 

# Raison pour laquelle, nous n'aurons meme pas besoin de refaire la standarisation de tous dataframes et des données, mais plutot on s'intéresse à la transformation de Time et Amount.

# In[66]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(dt.values)
print(X[0])


# In[69]:



dt['Amount'] = scaler.fit_transform(dt['Amount'].values.reshape(-1, 1))
dt.drop('Time', axis=1, inplace=True)

feature_columns = dt.columns.values.tolist()
feature_columns.remove('Class')
#target = 'Class'


# In[86]:


from sklearn.decomposition import PCA
Y=dt[feature_columns]
reduced = PCA(n_components=2).fit_transform(Y.values)

plt.figure()
plt.scatter(reduced[dt['Class'] == 0, 0], reduced[dt['Class'] == 0, 1], color='blue')
plt.scatter(reduced[dt['Class'] == 1, 0], reduced[dt['Class'] == 1, 1], color='red', marker='x')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title('Reduction des dimensions: PCA')


# Nous avons utilisé le PCA pour réduire la dimensionnalité de notre ensemble de données à 2 dimensions afin de pouvoir le tracer. Cela nous dira s'il y a une certaine séparation entre la classe positive et la classe négative.
# Le graphique nous montre les deux  composants principaux et nous pouvons voir que les transactions frauduleuses sont regroupées dans la partie inférieure gauche du graphique.

# In[74]:


dt.describe()


# In[75]:


dt['Class'].value_counts()


# On en déduit que notre dataset est toujours imbalancée

# In[76]:


sns.countplot(x=dt.Class, hue=dt.Class)
plt.title("Transaction Class Distribution : après réduction")
LABELS = ['Legit','Fraud']
plt.xticks(range(2), LABELS)


# L'idee est simple, pour travailler sur une dataset balancée , nous allons prendre un échantillon aléatoire de notre ensemble de données et nous concaténons 2 dataframes : si axis = 0 ce dataframe sera ajouté un par un. Et ceci est connu sous le nom du susample.
# EN effet le subsample (undersampling)est une technique permettant d’équilibrer les ensembles de données inégaux en conservant toutes les données dans la classe minoritaire et en diminuant la taille de la classe majoritaire. C’est l’une des nombreuses techniques que les Datascientist peuvent utiliser pour extraire des informations plus précises à partir d’ensembles de données initialement déséquilibrés.
# 
# Nous aurons alors des données distribuées uniformes avec 492 transactions frauduleuse et 492 transactions normales . c'est comme ceci que nous avons obtenu l'amélioration de notre modèle dans ce qui suit.

# In[89]:


legit_sample = LEGIT.sample(n=492)
new_dt = pd.concat([legit_sample, FRAUD], axis=0)

new_dt.head()


# La nouvelle dataset améliorée: new_dt

# In[90]:


X = new_dt.drop(columns='Class', axis=1)
Y = new_dt['Class']


# In[91]:


new_dt['Class'].value_counts()


# In[92]:


sns.countplot(x=new_dt.Class, hue=new_dt.Class)
plt.title("Transaction Class Distribution : après undersampling")
LABELS = ['Normal','Fraud']
plt.xticks(range(2), LABELS)


# Maintenant après avoir appliqué la technique du Undersampling, nous avons obtenu notre dataframe balancée correctement : 492 Fraud et 492 Legit 

# # Matrice de corrélation : 

# Les matrices de corrélation sont l'essence même de la compréhension de nos données. Nous voulons savoir s'il existe des caractéristiques qui influencent fortement si une transaction spécifique est une fraude.

# In[93]:


plt.subplots(figsize=(24,20))
sub_sample_corr = new_dt.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r',annot=True)
plt.show()


# Interprétation des résultats : 

# Corrélations négatives : V17, V14, V12,V16 et V10 sont corrélés négativement.

# Corrélations positives : V2, V4, V11 et V19,V21,V27 sont positivement corrélés.

# 

# Donc, meme si on a notre dataset standarisée de base, et on a essayé de réduire les dimensions encore avec PCA, ceci n'a pas aboutit a avoir une dataset balancée, vu que le PCA réduit seulement les colonnes.
# Alors on a balancé notre dataset avec la méthode la plus utilisée c'est le undersampling, afin d'avoir un modèle enfin du compte : standarisé, réduit(avec PCA) et balancée( avec undersampling)
Alors, voici notre modèle  qui nous garantira effectivement de bien visualiser les données réduites, et mieux détecter les cas Fraud des cas Legit.
# # Chi-square :

# Le Chi-square est un test d'hypothèse statistique valide lorsque la statistique de test est distribuée par le chi carré sous l'hypothèse nulle. C'est un test la dépendance entre les variables catégorielles.

# In[95]:


import researchpy as rp
import scipy.stats as stats


# In[94]:


new_dt.info()


# In[104]:


rp.summary_cat(new_dt[["Class", "Amount"]])


# IL faut chercher une relation entre : Class(Fraud ou Legit) Et Amount

# In[105]:


crosstab = pd.crosstab(new_dt["Class"], new_dt["Amount"])

crosstab


# On peut prédire la détection des fraudes ou pas avec meme la quantité des fraudes réalisés.

# In[106]:


stats.chi2_contingency(crosstab)

il y a une relation entre les distributions Class et Amount telle que : Pearson Chi-square(599)= 855.48, p<0.0001
# In[109]:


crosstab, test_results, expected = rp.crosstab(new_dt["Class"], new_dt["Amount"],
                                               test= "chi-square",
                                               expected_freqs= True,
                                               prop= "cell")

crosstab


# In[110]:


test_results


# Interprétation : 

# la valeur de : Cramer's V est = 0.9324 >0.8 , donc il y a une forte dépendance  entre Class et Amount par indice de V Cramer 
# 

# Mais pour le test du p-value, on ne peut pas conclure, on n'arrive pas à interpreter ce résultat.

# In[111]:


expected

Dans l'étape suivante, après avoir obtenu un modèle réduit des données, on arrive à l'étape majeure du Machine Learning, c'est le test des méthodes PMC et SVM :
# # Etape 4 : Machine Learning (Algorithmes : PMC ET SVM) 

# Cette étape consiste à tourner les algorithmes PMC et SVM afin de pouvoir choisir le meilleur modèle pour la représentation des donnéSes : 

# # 1) Perceptron Multicouche : PMC 

# In[112]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)


# In[113]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(activation='tanh', solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(13), max_iter=1000)
print("Sans Cross-Validation: ")
mlp.fit(X_train,Y_train)
print("Training set score: %f" % mlp.score(X_train, Y_train))
print("Test set score: %f" % mlp.score(X_test, Y_test))
print("avec Cross-Validation: ")
kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
cv_results = cross_val_score(mlp, X, Y, cv=kfold, scoring='accuracy')
print('MLP: %f' % (cv_results.mean()))

Matrice de confusion pour : Perceptron multicouche(PMC)
# In[114]:


mlp_pred_train = mlp.predict(X_train)
mlp_pred_test = mlp.predict(X_test)

tn, fp, fn, tp = confusion_matrix(Y_test, mlp_pred_test).ravel()
conf_matrix = pd.DataFrame(
    {
        'Predicted Fraud': [tp, fp],
        'Predicted Not Fraud': [fn, tn]
    }, index=['Fraud', 'Not Fraud'])
conf_matrix


# In[115]:


sns.heatmap(conf_matrix, annot=True)


# # 2) Support Vector Machines

# In[116]:


from sklearn.svm import SVC
svc = SVC(gamma='auto')
print("Sans Cross-validation")
svc.fit(X_train, Y_train)
print("Training set score: %f" % svc.score(X_train, Y_train))
print("Test set score: %f" % svc.score(X_test, Y_test))
print("Avec Cross-validation")
kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
cv_results = cross_val_score(svc, X, Y, cv=kfold, scoring='accuracy')
print('SVC: %f' % (cv_results.mean()))

Matrice de confusion pour : Support Vector Machine (SVM)
# In[117]:


svc_pred_train = svc.predict(X_train)
svc_pred_test = svc.predict(X_test)

tn, fp, fn, tp = confusion_matrix(Y_test, svc_pred_test).ravel()
conf_matrix = pd.DataFrame(
    {
        'Predicted Fraud': [tp, fp],
        'Predicted Not Fraud': [fn, tn]
    }, index=['Fraud', 'Not Fraud'])
conf_matrix


# In[118]:


sns.heatmap(conf_matrix, annot=True)


# # 5) Validation des modèles :

# Dans cette dernière étape, nous devons choisir lequel des modèles traités est meilleur pour la détection des Fraudes dans notre Dataset : creditcard.csv 

# Alors comment peut-on sélectionner le meilleur moèele?selon quels critères ? 

# D'après avoir calculé les valeurs estimées du PMC ET SVM avec et sans cross validation, et en s'appuyant également sur la matrice de confusion, on peut sélectionner le modèle qui a la valeur de validation la plus proche à 1  (dans le cas de Cross Validation):
# 

# # Perceptron multicouche: 0.627344  (cross validation)

# # Support vecteur machine : 0.548778 ( cross validation )

# 
# 

# 

# # Le meilleur modèle est : Le perceptron Multicouche.

# # Conclusion : Pour le cas de notre dataset avec des données bien spécifiques, nous avons trouvé que La méthode du PMC est plus optimale que SVM pour la détection des fraudes des cartes de crédits.

# 
Fin de réalisation . Merci 