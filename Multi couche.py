import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt



t = time.time()
train_data = pd.read_csv('C:/Users/Utilisateur/Documents/L3 TSE/S5/Projet informatique magistère/mnist_train.csv')
test_data = pd.read_csv("C:/Users/Utilisateur/Documents/L3 TSE/S5/Projet informatique magistère/mnist_test.csv")
#train_data = pd.read_csv('C:/Users/ghani/Documents/Desktop/TSE/Licence 3/S1/Projet Magistere/mnist_train.csv')
#test_data = pd.read_csv('C:/Users/ghani/Documents/Desktop/TSE/Licence 3/S1/Projet Magistere/mnist_test.csv')

#X_train=train_data.drop(labels=["label"],axis=1)
#Y_train=train_data["label"]

#X_test=test_data.drop(labels=["label"],axis=1)
#Y_test=test_data["label"]

X_train = train_data.drop(labels=["label"], axis=1).to_numpy()  # convertir en numpy array
Y_train = train_data["label"].to_numpy()
X_test = test_data.drop(labels=["label"], axis=1).to_numpy()
Y_test = test_data["label"].to_numpy()

X_train = X_train / 255.0 #on normalise
X_test = X_test / 255.0

print(time.time() - t)

#Une classe Deux couches

import numpy as np


class MLP:
    def __init__(self, taille_couches, tx_app):

        self.taille_couches = taille_couches  # liste de dim par couche
        self.tx_app = tx_app
        self.L = len(taille_couches) - 1  # Nombre de couches (sans la couche d'entrée)
        self.poids = []
        self.biais = []

        for l in range(self.L):
            input_sz = taille_couches[l]
            output_sz = taille_couches[l+1]
            self.poids.append(np.random.randn(input_sz, output_sz) * 0.01)
            self.biais.append(np.random.randn(output_sz) * 0.01)

    def activation_function(self, v):
        return 1 / (1 + np.exp(-v))

    def activation_derivative(self, a):
        return a * (1 - a)

    def softmax(self, v):
        # Ensure v is a 2D array
        if v.ndim == 1:
            v = v.reshape(1, -1)
        return np.exp(v) / np.sum(np.exp(v), axis=1, keepdims=True)

    def forward_propagation(self, M):
        activations = [M] #stocke les activations des couches
        zs = []  #stocke les valeurs avant activation

        a = M
        for l in range(self.L-1): #j'ajoute -1 parce que la derniere se fait avec softmax dans prediction
            v = np.dot(a, self.poids[l]) + self.biais[l]
            zs.append(v)
            a = self.activation_function(v)
            activations.append(a)

        v_final = np.dot(a, self.poids[-1]) + self.biais[-1]
        a_final = self.softmax(v_final)  #on applique softmax a la derniere couche
        activations.append(a_final)
        zs.append(v_final)  #stockage de la dernière somme pondérée

        return a_final, activations, zs

        #a represente la sortie finale du reseau(avant softmax), vecteur de taille [1,10]
        #activations liste avec toutes les activations des couches
        #zs liste avec toutes les valeurs avant activations
        return a,activations,zs

    def prediction (self,I):
        return self.softmax(self.forward_propagation(I)[0])

    def back_propagation (self,X,Y):
        m=X.shape[0]
        activations, zs = self.forward_propagation(X)[1], self.forward_propagation(X)[2]

        #Matrice des gradients, matrice mm forme mais avec zeros
        d_poids = [np.zeros_like(w) for w in self.poids]
        d_biais = [np.zeros_like(b) for b in self.biais]

        #On transforme une valeur catégorielle en vecteur binaire
        Y_one_hot = np.zeros((m, self.taille_couches[-1]))
        Y_one_hot[np.arange(m), Y] = 1

        #on calcul du gradient de la perte pour la derniere couche
        #c'est la derivee de l'entropie croisée avec softmax
        delta = activations[-1] - Y_one_hot

        #Backpropagation
        #propagation du gradient aux poids et biais
        for l in range(self.L-1,-1,-1):
            d_poids[l] = np.dot(activations[l].T, delta)/m
            d_biais[l] = np.mean(delta, axis=0)
            #tant qu'on atteint pas la derniere couche
            if l > 0:
                delta = np.dot(delta, self.poids[l].T) * self.activation_derivative(activations[l])

        # Mise à jour des poids et biais avec descente de gradient
        for l in range(self.L):
            self.poids[l] -= self.tx_app*d_poids[l]
            self.biais[l] -= self.tx_app*d_biais[l]




taille= [784,150,250,10]  # liste de dim par couche
taux_app=0.1
print("aaaaa")
P = MLP (taille, taux_app)

test = [(X_test[i], Y_test[i]) for i in range(10000)]
I = test[0][0]
prediction = P.prediction(I)
print(prediction)


# Paramètres d'entraînement
#on va utiliser des mini-lot, "batch" pour eviter d'utiliser toute la base de donnees d'un coup
#et aussi pour genre accelerer l'entrainement comme c'est deja assez long
#on utilise des sous-ensemble du dataset pour entrainer le reseau
#sinon ca prend trop de memoire et c'est trop lent
n_epochs = 10  # Nombre de fois où on passe sur tout le dataset
#une epoque passe sur toutes les images, mais en petits groupes
batch_size = 68  # Nombre d'images utilisées à chaque mise à jour des poids

for epoch in range(n_epochs):
    print(f"Epoch {epoch+1}/{n_epochs}")

    indices = np.arange(len(X_train))
    np.random.shuffle(indices)#mélange des données
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        #image de i à i+batch_size
        Y_batch = Y_train[i:i + batch_size]
        #prend les labels correspondants

        P.back_propagation(X_batch, Y_batch)  # Mise à jour des poids avec backpropagation

    print("Fin de l'époque", epoch + 1)


I = X_test[7]  # Une image test
prediction = P.prediction(I)
print("Prédiction :", np.argmax(prediction))  # Classe prédite
print("Vraie étiquette :", Y_test[7])  # Classe réelle