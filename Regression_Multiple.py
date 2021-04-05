import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

data = pd.read_csv('boston_house_prices.csv')
data.columns

X = data.drop('MEDV', axis=1)
Y = data.MEDV

# Affichons le nuage de points de chacunes des colonnes de notre dataset
# en fonction de celle target (MEDV)

"""
data["CRIM"]

plt.scatter(data["CRIM"], Y)
plt.show()
"""

plt.figure(figsize=(75,5))
for i, col in enumerate(X.columns):
    plt.subplot(1, 13, i+1)
    x = data[col]
    #y = Y
    plt.plot(x, Y, 'o')
    # Création de la ligne de regression
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, Y, 1))(np.unique(x)))
     
    plt.style.use(['dark_background', 'fast'])
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('Prix')
    
#------------------------------------------------------------------------------
#Fractionnement du dataset entre le trainning set et le Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

scaler = StandardScaler()

scaler.fit(X_train)
scaler.fit(X_test)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Construction du modèle
regressor = LinearRegression()

# J'adapte le modèle de regression linéaire à l'ensemble de données d'apprentissage.
regressor.fit(X_train, Y_train)

# Faire de nouvelles prédictions
y_pred = regressor.predict(X_test)

# Visualisation
plt.style.use("bmh")
plt.scatter(y_pred, Y_test)
plt.show()

regressor.predict(scaler.fit_transform(np.array([[0.17331, 0, 9.69, 0, 0.585, 5.707, 54, 2.3817, 6, 391, 19.2, 396.9, 12.01]])))

#============================= Evaluation et Validation ===============================

constante = regressor.intercept_
print("Constante : ", constante)

coefficients = regressor.coef_
print("Coefficient : ", coefficients)

erreur_quadratique_moyenne = np.mean((y_pred - Y_test)**2)
print("L'erreur Quadratique Moyenne : ", erreur_quadratique_moyenne)

[i for i in list(X)]
[i for i in enumerate(X.columns)]

nom = [i for i in list(X)]

import statsmodels.api as sm

# On fait appel à OLS qui va permettre de recuperer le résumé statistique de tous les éléments

model = sm.OLS(Y_train, X_train)
result = model.fit()

print(result.summary())

#=== On crée un deuxieme modele pour le confronter au premier 
# le deuxieme modele est obtenu en retirant la colonne age des futures
#=============================== Model 2 =====================================

X1 = X[['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']].values

Y1 = Y

#============ FRACTIONNEMENT DU DATASET ENTRE LE TRAINNING SET ET LE TEST SET ==============
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.2, random_state=0)

scaler1 = StandardScaler()
scaler1.fit(X1_train)
scaler1.fit(X1_test)

X1_train, X1_test = scaler1.fit_transform(X_train), scaler1.fit_transform(X_test)

# Construction du modèle
regressor1 = LinearRegression()

# J'adapte le modèle de regression linéaire à l'ensemble de données d'apprentissage.
regressor1.fit(X1_train, Y1_train)

# Faire de nouvelles prédictions
y_pred1 = regressor1.predict(X1_test)

# Visualisation
plt.style.use("bmh")
plt.scatter(y_pred1, Y1_test)
plt.show()

#======================= Evaluation ===========================
erreur_quadratique_moyenne = np.mean((y_pred1 - Y1_test)**2)
print("L'erreur Quadratique Moyenne : ", erreur_quadratique_moyenne)

model1 = sm.OLS(Y1_train, X1_train)
result = model.fit()
print(result.summary())

#================== Faire face à la multicolinéarité ====================



