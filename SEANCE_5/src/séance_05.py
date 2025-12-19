# -*- coding: utf-8 -*-

import pandas as pd
import math
import scipy
import scipy.stats
import numpy as np

#Fonction locale
def ouvrirUnFichier(nom):
    with open(nom, "r") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

#POP MERE
pop_mere_tot = 2185
pop_mere_pour = 852
pop_mere_contre = 911
pop_mere_sans_opinion =  422

#THEORIE DE L'ECHANTILLONAGE
#Importation du jeu de données sur les échantillons
donnees = pd.DataFrame(ouvrirUnFichier("data/Echantillonnage-100-Echantillons.csv"))

sum(list(donnees.iloc[0]))

#Calcule de la moyenne de chaque opinion
df = pd.DataFrame(donnees)

moyenne_pour = round(df['Pour'].mean(),0)
moyenne_contre = round(df['Contre'].mean(),0)
moyenne_sans_opinion = round(df['Sans opinion'].mean(),0)

#Calcule de la fréquence de l'échantillon
# 1- somme des moyennes obtenues
nombres = [moyenne_pour, moyenne_contre, moyenne_sans_opinion]
somme = sum(nombres)

# 2- Moyenne de chaque opinion divisée par la somme des moyennes
pour = moyenne_pour / somme
contre = moyenne_contre / somme
sans_opinion = moyenne_sans_opinion / somme

print("Fréquence pour l'opinion de l'échantillon :")
print (f"       - Pour : {pour:.2f}")
print (f"       - Contre : {contre:.2f}")
print (f"       - Sans opinion : {sans_opinion:.2f}")

#Calcule de la fréquence de la population mère
pour_mere = round(pop_mere_pour / pop_mere_tot,2)
contre_mere = round(pop_mere_contre / pop_mere_tot,2)
sans_opinion_mere = round(pop_mere_sans_opinion / pop_mere_tot,2)

print("Fréquence pour l'opinion de la population mère :")
print (f"       - Pour : {pour_mere}")
print (f"       - Contre : {contre_mere}")
print (f"       - Sans opinion : {sans_opinion_mere}")

#Calcule de l'intervale de fluctuation
def intervalle_fluctuation(p, n, seuil=0.95):
    if not (0 < p < 1):
        raise ValueError("La proportion p doit être entre 0 et 1.")
    if n <= 0:
        raise ValueError("La taille de l'échantillon n doit être positive.")

    # z pour un intervalle à 95 %, z ≈ 1.96
    z = 1.96 if seuil == 0.95 else math.sqrt(2)
    marge = z * math.sqrt((p * (1 - p)) / n)

    return (p - marge, p + marge)

#Intervale de fluctuation seuil 95%
for i in [pour,contre,sans_opinion]:
  print(intervalle_fluctuation(i, 1000))

print(f"Résultat sur le calcul d'un intervalle de fluctuation: {intervalle_fluctuation(i, 1000)}")

#THEORIE DE L'ESTIMATION (intervalles de confiance)
#Sélection de la première ligne avec iloc[0]
premier_echantillon = donnees.iloc[0]

#Conversion en liste Python
premier_echantillon_liste = list(premier_echantillon)

print("\nPremier échantillon converti en liste:", premier_echantillon_liste)

#Calcul de la somme de la ligne
somme_ligne = sum(premier_echantillon)

#Calcul des fréquences
frequences = [val / somme_ligne for val in premier_echantillon]
print("Fréquences :", frequences)

#Isoler la première ligne
echantillon = donnees.iloc[0]

#Taille de l'échantillon
n = echantillon.sum()

#Calcul des fréquences et IC pour chaque opinion
resultats = {}
for opinion, valeur in echantillon.items():
    p = valeur / n
    resultats[opinion] = {
        "frequence": p,
        "IC95": (intervalle_fluctuation(p, n))
    }

print("Taille de l'échantillon :", n)
for opinion, infos in resultats.items():
    print(f"Intervalle de confiance : {opinion} : fréquence = {infos['frequence']:.3f}, IC95 = {infos['IC95']}")

#THEORIE DE LA DECISION (tests d'hypothèse)
#La décision se base sur la notion de risques alpha et bêta.

#Importation de nos deux jeux de données
loi_1 = pd.DataFrame(ouvrirUnFichier("data/Loi-normale-Test-1.csv"))
loi_2 = pd.DataFrame(ouvrirUnFichier("data/Loi-normale-Test-2.csv"))

valeurs1 = loi_1["Test"]
valeurs2 = loi_2["Test"]

# Test de Shapiro-Wilk
stat1, p1 = scipy.stats.shapiro(valeurs1)
stat2, p2 = scipy.stats.shapiro(valeurs2)

print("Test Shapiro-Wilk pour Loi-normale-Test-1 :")
print("Statistique =", stat1, "p-value =", p1)
print(f"La distribution est {'Non normale' if p1 <= 0.05 else 'Normale'}")

print("\nTest Shapiro-Wilk pour Loi-normale-Test-2 :")
print("Statistique =", stat2, "p-value =", p2)
print(f"La distribution est {'Non normale' if p2 <= 0.05 else 'Normale'}")
