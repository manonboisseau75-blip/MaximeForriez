# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.stats import spearmanr, kendalltau
import seaborn as sns
import math
import plotly.express as px

#Fonction pour ouvrir les fichiers
def ouvrirUnFichier(nom):
    return pd.read_csv(nom)

#Fonction pour convertir les données en données logarithmiques
def conversionLog(liste):
    return np.log(liste)

#Fonction pour trier par ordre décroissant les listes (îles et populations)
def ordreDecroissant(liste):
    liste.sort(reverse = True)
    return liste

#Fonction pour obtenir le classement des listes spécifiques aux populations
def ordrePopulation(pop, etat):
    ordrepop = []
    for element in range(0, len(pop)):
        if np.isnan(pop[element]) == False:
            ordrepop.append([float(pop[element]), etat[element]])
    ordrepop = ordreDecroissant(ordrepop)
    for element in range(0, len(ordrepop)):
        ordrepop[element] = [element + 1, ordrepop[element][1]]
    return ordrepop

#Fonction pour obtenir l'ordre défini entre deux classements (listes spécifiques aux populations)
def classementPays(ordre1, ordre2):
    classement = []
    if len(ordre1) <= len(ordre2):
        for element1 in range(0, len(ordre2) - 1):
            for element2 in range(0, len(ordre1) - 1):
                if ordre2[element1][1] == ordre1[element2][1]:
                    classement.append([ordre1[element2][0], ordre2[element1][0], ordre1[element2][1]])
    else:
        for element1 in range(0, len(ordre1) - 1):
            for element2 in range(0, len(ordre2) - 1):
                if ordre2[element2][1] == ordre1[element1][1]:
                    classement.append([ordre1[element1][0], ordre2[element2][0], ordre1[element1][1]])
    return classement

#PARTIE SUR LES ILES
#Importation du jeu de données sur les iles
iles = ouvrirUnFichier("data/island-index.csv")

#Isolation de la colonne "Surface (km2)"
iles['Surface (km²)']

#On convertit nos données en "float"
(np.array(list(iles['Surface (km²)']) + [85543323.0,37856841.0,7768030.0,7605049.0])).astype(float)

#On classe nos données par ordre décroissant
taille_ord = ordreDecroissant((list(iles['Surface (km²)']) + [85543323.0,37856841.0,7768030.0,7605049.0]))

#Visualisation de la loi rang-taille
plt.plot(range(len(taille_ord)),conversionLog(taille_ord))
plt.show()
# On peut surement faire des tests mais je vois pas ce que vous voulez dire

#PARTIE SUR LES POPULATIONS DES ETATS DU MONDE
#Source. Depuis 2007, tous les ans jusque 2025, M. Forriez a relevé l'intégralité du nombre d'habitants dans chaque États du monde proposé par un numéro hors-série du monde intitulé États du monde. Vous avez l'évolution de la population et de la densité par année.
monde = (ouvrirUnFichier("data/Le-Monde-HS-Etats-du-monde-2007-2025.csv"))

#On isole certaines de nos colonnes
monde[['État','Pop 2007','Pop 2025','Densité 2007','Densité 2025']].sort_values('Pop 2007',ascending=False).reset_index(drop=True)

#On range nos listes dans l'ordre décroissant
pop2007 = ordrePopulation(monde['Pop 2007'], monde['État'])
pop2025 = ordrePopulation(monde['Pop 2025'], monde['État'])
densite2007 = ordrePopulation(monde['Densité 2007'], monde['État'])
densite2025 = ordrePopulation(monde['Densité 2025'], monde['État'])

p2007 = list(classementPays(pop2007,densite2007))

np.sort(list(classementPays(pop2025,densite2025)))

#On isole nos colonnes 2007 en utilisant une boucle
pop2007_rangs, dens2007_rangs = [], []
for i in p2007:
  pop2007_rangs.append(i[0])
  dens2007_rangs.append(i[1])

#Coefficient de corrélation
print(px.scatter(x=pop2007_rangs, y=dens2007_rangs, trendline="lowess", trendline_options=dict(frac=0.8)))

#Méthode Spearmanr
print(spearmanr(pop2007_rangs, dens2007_rangs))

#Méthode Kendalltau
print(kendalltau(pop2007_rangs, dens2007_rangs))