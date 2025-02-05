import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / 1 - np.exp(-x)

def sigmoid_deriv(x):
    return x * (1 - x)

def produitmat(A,B):
    return np.dot(A, B)



t = time.time()
train_data = pd.read_csv('C:/Users/Utilisateur/Documents/L3 TSE/Projet informatique magistère/mnist_train.csv')
test_data = pd.read_csv("C:/Users/Utilisateur/Documents/L3 TSE/Projet informatique magistère/mnist_test.csv")
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
class DeuxCouches():

    def __init__(self, input_size, output_size,a ,b learning_rate=0.1):
        np.random.seed(42)
        self.learning_rate = learning_rate
        self.w = []
        self.biais = []
        self.a = a

        #Initialisation en trois fois :
        #couche d'entrée vers la première couche cachée
        #couches cachées
        #dernière couche cachée vers la sortie
        self.w.append(np.random.randn(b, input_size))
        self.biais.append(np.zeros((b, 1)))

        for i in range(a - 1):
            self.w.append(np.random.randn(b, b))
            self.biais.append(np.zeros((b, 1)))

        self.w.append(np.random.randn(output_size, b))
        self.biais.append(np.zeros((output_size, 1)))


    def backward(self, X, y):
        m = X.shape[0]
        erreurs = [None] * (self.a + 1)  # Liste pour stocker les erreurs

        #Erreur en sortie
        dA = (self.activations[-1] - y.T) * sigmoid_deriv(self.activations[-1])
        erreurs[-1] = dA #donne l'erreur pour la derniere couche

        #retropropagation des erreurs
        for l in range(self.a, 0, -1):
            erreurs[l - 1] = np.dot(self.w[l].T, erreurs[l]) * sigmoid_deriv(self.activations[l])

        #mise à jour des poids et biais
        for l in range(len(self.w)):
            dW = np.dot(erreurs[l], self.activations[l].T) / m
            db = np.sum(erreurs[l], axis=1, keepdims=True) / m
            self.w[l] -= self.learning_rate * dW
            self.biais[l] -= self.learning_rate * db





