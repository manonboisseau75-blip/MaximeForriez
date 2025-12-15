import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import linregress, pearsonr
from sklearn.linear_model import LinearRegression
import plotly.express as px

def ouvrirUnFichier(nom):
    with open(nom, "r") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

data = pd.DataFrame(pd.read_csv("data/pib-vs-energie.csv"))

#Sélection des données de 1991, les données de 2022 comme demandées dans les consignes ne sont pas présentes
data_trans = data[['PIB_1991', 'Utilisation_d_energie_1991']]
data_trans_float = data_trans.astype(float)

#Création d'un algorithme pour mettre de côté toutes les données qui sont manquantes
def suppr_incomplet(df, champ_x, champ_y):
  couples_complets = []
  for i in range(len(df)):
    # Vérifier que les deux valeurs ne sont pas NaN
    x = df[champ_x][i]
    y = df[champ_y][i]
    if not np.isnan(x) and not np.isnan(y):
        couples_complets.append((x, y))
  return pd.DataFrame(couples_complets, columns=[champ_x, champ_y])

#Utilisation de l'algorithme
couples_complets = suppr_incomplet(data_trans_float, 'PIB_1991','Utilisation_d_energie_1991')
print("Couples complets (PIB, Conso) :")
print(couples_complets)

#Calcul de la régression linéaire pour la méthode des moindres carrées
print("Calcul de la régression linéaire pour la méthode des moindres carrées")
regression = linregress(couples_complets['PIB_1991'],couples_complets['Utilisation_d_energie_1991'])
print(regression)
print()

#Calcul du coefficient de corrélation simple de Pearson
print("Calcul du coefficient de corrélation simple de Pearson")
print(pearsonr(couples_complets['PIB_1991'],couples_complets['Utilisation_d_energie_1991']))

#Graphique de synthèse
plt.plot(couples_complets['PIB_1991'],couples_complets['Utilisation_d_energie_1991'], 'o')
x = couples_complets['PIB_1991'].values.reshape([-1,1])
model = LinearRegression()
model.fit(x, couples_complets['Utilisation_d_energie_1991'])

fig = px.scatter(couples_complets, x='PIB_1991', y='Utilisation_d_energie_1991',trendline='ols')
fig.show()
