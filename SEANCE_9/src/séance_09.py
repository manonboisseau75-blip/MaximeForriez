#coding:utf8

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import scipy
import scipy.stats
from scipy.cluster.hierarchy import dendrogram, linkage
import prince
import matplotlib.pyplot as plt
import os

def ouvrirUnFichier(nom):
    with open(nom, "r") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

#Création du dossier img
try:
  os.mkdir('img')
except:
  pass

#Analyse en composantes principales
temperature =ouvrirUnFichier("data/france-temperatures.csv")

# Isoler la colonne des individus "Villes"
villes = temperature['Villes']

# Isoler les données numériques
X = temperature.drop(columns=["Villes"])

# Retirer la colonne "Villes" pour ne garder que les données numériques
donnees_numeriques = temperature.drop(columns=['Villes'])

# Centrer-réduire les données numériques
scaler = StandardScaler()
donnees_standardisees = scaler.fit_transform(donnees_numeriques)

# Convertir le résultat en DataFrame
donnees_standardisees_df = pd.DataFrame(donnees_standardisees, columns=donnees_numeriques.columns)
print(donnees_standardisees_df)

# Créer l'objet PCA avec 12 facteurs
pca = PCA(n_components=12)
X_scaled = scaler.fit_transform(X)

# Appliquer l'ACP
X_pca = pca.fit_transform(X_scaled)

# Obtenir les composantes principales
donnees_pca = pca.transform(donnees_standardisees)

# Convertir en DataFrame pour visualiser
donnees_pca_df = pd.DataFrame(donnees_pca, columns=[f'Facteur_{i+1}' for i in range(12)])
print(donnees_pca_df)

print("Variance expliquée par composante :")
print(pca.explained_variance_ratio_)

# Variance expliquée cumulée
print("Variance expliquée cumulée :")
print(pca.explained_variance_ratio_.cumsum())

# Valeurs propres
valeurs_propres = pca.explained_variance_

# Variance expliquée en pourcentage
variance_pourcentage = pca.explained_variance_ratio_ * 100

# Variance cumulée
variance_cumulee = variance_pourcentage.cumsum()

# Créer le tableau
tableau_acp = pd.DataFrame({
    'Facteur': [f'Facteur {i+1}' for i in range(len(valeurs_propres))],
    'Valeur propre': valeurs_propres,
    'Variance expliquée (%)': variance_pourcentage,
    'Variance cumulée (%)': variance_cumulee
})
print(tableau_acp)

# Affichage avec un DataFrame avec les composantes principales
resultat_acp = pd.DataFrame(X_pca, columns=[f'Facteur {i+1}' for i in range(X_pca.shape[1])])
print(resultat_acp)

#création du graphique
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], color='blue')

#ajout des villes
villes = temperature['Villes']
for i, ville in enumerate(villes):
    plt.text(X_pca[i, 0], X_pca[i, 1], ville, fontsize=9)

#juste les deux premiers facteur
plt.xlabel('Facteur 1')
plt.ylabel('Facteur 2')
plt.title("Mapping des individus selon les deux premiers facteurs de l'ACP")
plt.grid(True)
plt.savefig("img/mapping_ACP.png")

# Valeurs propres
eigenvalues = pca.explained_variance_

n = X_scaled.shape[0]  # nombre d'individus

# Contribution des individus
contrib = (X_pca**2) / (n * eigenvalues)

contrib_df = pd.DataFrame(contrib,
                          index=villes,
                          columns=[f"Dim{i+1}" for i in range(X_pca.shape[1])])

print("Contribution des individus :")
print(contrib_df.head())

#Qualité de la projection (cos²)
# Norme au carré de chaque individu
normes = np.sum(X_pca**2, axis=1)

cos2 = (X_pca**2) / normes[:, np.newaxis]

cos2_df = pd.DataFrame(cos2,
                       index=villes,
                       columns=[f"Dim{i+1}" for i in range(X_pca.shape[1])])

print("\nQualité de la projection (cos²) :")
print(cos2_df.head())

#Coordonnées des variables (loadings)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

#CERCLE DE CORRELATION
fig, ax = plt.subplots(figsize=(6,6))

# Cercle unité
circle = plt.Circle((0,0), 1, color='blue', fill=False)
ax.add_artist(circle)

# Coordonnées des variables sur Dim1 et Dim2
xs = loadings[:,0]
ys = loadings[:,1]
for i, var in enumerate(X.columns):
    ax.text(xs[i], ys[i], var)

ax.scatter(xs, ys)
ax.axhline(0, color='grey', linestyle='--')
ax.axvline(0, color='grey', linestyle='--')
ax.set_xlabel("Dim1")
ax.set_ylabel("Dim2")
ax.set_title("Cercle de corrélation")

plt.xlim(-1, 1)
plt.ylim(-1, 1)

# Sauvegarde dans le dossier img
plt.savefig("img/cercle_correlation.png")

#REALISATION ACM
chien = ouvrirUnFichier("data/chiens.csv")

#Garde toutes les colonnes sauf "Race"
variables = chien.drop(columns=["Race"])

# Transformer en tableau disjonctif complet
tdc = pd.get_dummies(variables)
print(tdc)

# Calculer l'ACM avec 8 facteurs
mca = prince.MCA(
    n_components=8,
    n_iter=10,
    copy=True,
    check_input=True,
    engine='sklearn',
    random_state=42
)

# Ajuster le modèle
mca = mca.fit(tdc)

# Coordonnées des individus
coords_individus = mca.row_coordinates(tdc)

# Coordonnées des variables
coords_variables = mca.column_coordinates(tdc)

# Valeurs propres
eigenvalues = mca.eigenvalues_

# Inertie expliquée par axe (proportion)
explained_inertia = eigenvalues / np.sum(eigenvalues)

print("Coordonnées des individus :\n", coords_individus)
print("\nCoordonnées des variables :\n", coords_variables)
print("\nValeurs propres :\n", eigenvalues)
print("\nInertie expliquée :\n", explained_inertia)

#creation de l'image
# Plot du mapping des deux premiers axes
plt.figure(figsize=(10, 8))

# Individus
plt.scatter(coords_individus[0], coords_individus[1], alpha=0.6, label="Individus")

# Modalités
plt.scatter(coords_variables[0], coords_variables[1], color="red", marker="x", label="Modalités")

plt.axhline(0, color="grey", lw=1)
plt.axvline(0, color="grey", lw=1)
plt.xlabel("Facteur 1")
plt.ylabel("Facteur 2")
plt.title("Mapping des deux premiers facteurs (A.C.M.)")
plt.legend()

# Sauvegarde dans le dossier img
plt.savefig("img/mapping_ACM.png")

#calcule des cos2
# Fonction pour calculer les similarités cosinus
def cosine_similarity_matrix(X):
    # Normalisation des vecteurs
    normed = X / np.linalg.norm(X, axis=1)[:, None]
    # Produit scalaire normalisé
    return np.dot(normed, normed.T)

# Matrice des similarités cosinus entre individus
row_cos2 = cosine_similarity_matrix(coords_individus.values)

col_cos2 = cosine_similarity_matrix(coords_variables.values)

print("Qualité de représentation des lignes (cos²) :")
print(row_cos2)

print("\nQualité de représentation des colonnes (cos²) :")
print(col_cos2)

#BONUS
Z = linkage(X_pca, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.savefig('img/dendogram.png')
plt.show()