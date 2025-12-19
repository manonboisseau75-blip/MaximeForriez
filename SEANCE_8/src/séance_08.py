#coding:utf8

import numpy as np
import pandas as pd
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway
from sklearn.preprocessing import MinMaxScaler
import prince

#Définition des fonctions
def ouvrirUnFichier(nom):
    with open(nom, "r",encoding='utf8') as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

def tableauDeContingence(nom, donnees):
    indexValeurs = {}
    for element in range(0,len(nom)):
        indexValeurs.update({element: nom[element]})
    return pd.DataFrame(donnees).rename(index = indexValeurs)

def sommeDesColonnes(tableau):
    colonne = list(tableau.head(0))
    sommeColonne = []
    for element in colonne:
        sommeColonne.append(tableau[element].sum())
    return sommeColonne

def sommeDesLignes(tableau):
    colonne = list(tableau.head(0))
    sommeLigne = []
    for element1 in range(0,len(tableau)):
        ligne = []
        for element2 in range(0,len(colonne)):
            ligne.append(tableau.iloc[element1, element2])
        sommeLigne.append(np.sum(list(ligne)))
    return sommeLigne

#Importation de la donnée
data = pd.DataFrame(ouvrirUnFichier("data/Socioprofessionnelle-vs-sexe.csv"))

#Création du tableau de contingence
contingence = tableauDeContingence(data["Catégorie"], {"Femmes": data["Femmes"], "Hommes": data["Hommes"]})
print(contingence)

#Isoler les colonnes
tableau = contingence[['Femmes','Hommes']].reset_index(drop=True)

#Calculer les lignes
sommeDesColonnes = (tableau)

#Calculer les marges
sommeDesLignes = (tableau)

#Faire une condition vérifiant si le total des marges des lignes et le total des marges des colonnes est identique
total_lignes = sommeDesLignes.sum().reset_index()
total_colonnes = sommeDesColonnes.sum().reset_index()

# Merge des données afin de pouvoir les 'aligner' et supprimer les lignes non présentes dans l'un des tableau (s'il y en a)
merged = pd.merge(total_lignes,total_colonnes,on='index')
if (merged['0_x'] - merged['0_y']).all():
    print("\nLes totaux des marges lignes et colonnes sont différents.")
else:
    print("\nLes totaux des marges lignes et colonnes sont identiques.")

tableau

#Test du chi2
chi2, p_value, dof, expected = chi2_contingency(tableau)
expected = pd.DataFrame(expected, index=tableau.index, columns=tableau.columns)

print("\nRésultats du test χ² :")
print("Statistique χ² :", chi2)
print("Degrés de liberté :", dof)
print("p-value :", p_value)
print("\nTableau des effectifs attendus :")
print(expected)

#Calculer l'intensité de liaison phi2 de Pearson
# Total des observations
n = tableau.to_numpy().sum()

# Coefficient de contingence (C de Pearson)
C = np.sqrt(chi2 / (chi2 + n))

print("Coefficient de contingence (C de Pearson) = intensite de liaison:", C)

def contribution_chi2(df,expected, labels:list, normalize: bool = True):
  # Contribution de chaque cellule
  contrib = (df - expected)**2 / expected

  # Convertir en DataFrame pour seaborn
  contrib_df = pd.DataFrame(contrib,
                            columns=['Femmes','Hommes'])
  if normalize:
    scaler = MinMaxScaler()
    contrib_df = pd.DataFrame(scaler.fit_transform(contrib_df),
                               columns=contrib_df.columns)

  # Heatmap
  plt.figure(figsize=(6,4))
  sns.heatmap(contrib_df, annot=True, cmap="Reds", cbar_kws={'label': 'Contribution au Chi²'}, yticklabels=labels)
  plt.title("Contribution des cellules au Chi²")
  plt.show()

if p_value < 0.05:
  contribution_chi2(tableau,expected,list(contingence.index))

#BONUS
for_anova = pd.read_csv('https://raw.githubusercontent.com/MaximeForriez/Sorbonne-M1-Analyse-de-donnees/refs/heads/main/Seance-05/Exercice/src/data/Echantillonnage-100-Echantillons.csv')

f_stat, p_val = f_oneway(for_anova['Pour'],for_anova['Contre'],for_anova['Sans opinion'])
print("F-statistique :", f_stat)
print("p-value :", p_val)

# Création de l'objet AFC
ca = prince.CA(n_components=2, n_iter=10, copy=True, check_input=True, engine='sklearn', random_state=42)

# Ajustement du modèle
ca = ca.fit(for_anova)

# Coordonnées des lignes
rows = ca.row_coordinates(for_anova)

# Coordonnées des colonnes
cols = ca.column_coordinates(for_anova)

plt.figure(figsize=(8,8))

# Lignes
plt.scatter(rows[0], rows[1], c='blue')

# Colonnes
plt.scatter(cols[0], cols[1], c='red')
for i, txt in enumerate(cols.index):
    plt.annotate(txt, (cols.iloc[i,0], cols.iloc[i,1]))

plt.axhline(0, color='grey', linestyle='--')
plt.axvline(0, color='grey', linestyle='--')
plt.xlabel('Axe 1')
plt.ylabel('Axe 2')
plt.title('AFC / CA - Plan factoriel')
plt.legend()
plt.show()