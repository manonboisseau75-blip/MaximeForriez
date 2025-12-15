#coding:utf-8

import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def ouvrirUnFichier(nom):
    with open(nom, "r") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

TEMPERATURE = "data/temperature.csv"
GEOMARKETING = "data/geomarketing.csv"

#Partie sur les températures
temperature = ouvrirUnFichier(TEMPERATURE)
temperature

#correlation
temperature = temperature.drop(columns=['Ville'])

correlation_matrix = temperature.corr()
print(correlation_matrix)

# Afficher les statistiques descriptives
print(correlation_matrix.describe())

# Isoler la variable à expliquer (Y)
Y = temperature['Temperature_en_janvier']
# Isoler les variables explicatives (X)
X = temperature[['Latitude', 'Longitude', 'Altitude']]

# Régression linéaire
X = sm.add_constant(X)

# Créer le modèle
model = sm.OLS(Y, X)

# Ajuster le modèle
results = model.fit()

print(results.summary())

#PARAMETRE DE REGRESSION
# 1. Les coefficients
print("Coefficients :\n", results.params)

# 2. Le coefficient de détermination R²
print("\nR² :", results.rsquared)

# 3. Les p-values
print("\nP-values :\n", results.pvalues)

#SKlearn
# Créer et ajuster le modèle
model = LinearRegression()
model.fit(X, Y)

# --- Résultats ---
# 1. Les coefficients
print("Coefficients :", model.coef_)

# 2. La constante (intercept)
print("Intercept :", model.intercept_)

#Partie sur le géomarketing
geomarketing = ouvrirUnFichier(GEOMARKETING)
geomarketing

#ISOLATION
# Variable à expliquer (cible)
Y_gm = geomarketing['ca']
# Variables explicatives significatives
X_gm = geomarketing[['surface_totale',
        'potentiel_Z20',
        'nb_primaire_Z10',
        'nb_primaire_Z20',
        'nb_gsa_Z10',
        'nb_pharmacie_Z5',
        'nb_conc2_Z10',
        'nb_conc2_Z20',
        'P10_POP_Z15',
        'P10_MEN_Z10']]

X

geomarketing_results = sm.OLS(Y_gm, X_gm).fit()
print(geomarketing_results.summary())

#PARAMETRE DE REGRESSION
# 1. Les coefficients
print("Coefficients :\n", geomarketing_results.params)

# 2. Le coefficient de détermination R²
print("\nR² :", geomarketing_results.rsquared)

# 3. Les p-values
print("\nP-values :\n", geomarketing_results.pvalues)