import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt



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

import numpy as np


class MLP:
    def __init__(self, couches_sizes, learning_rate):

        self.couches_sizes = couches_sizes  # liste de dim par couche
        self.learning_rate = learning_rate
        self.L = len(couches_sizes) - 1  # Nombre de couches (sans la couche d'entrée)

        self.weights = []
        self.biases = []

        for l in range(self.L):
            input_size = couches_sizes[l]
            output_size = couches_sizes[l + 1]
            self.weights.append(np.random.randn(input_size, output_size) * 0.01)
            self.biases.append(np.random.randn(output_size) * 0.01)

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
        activations = [M] #Stocke les activations des couches
        zs = []  #Stocke les valeurs avant activation

        a = M
        for l in range(self.L-1): #j'ajoute -1 parce que la derniere se fait avec softmax dans prediction
            v = np.dot(a, self.weights[l]) + self.biases[l]
            zs.append(v)
            a = self.activation_function(v)
            activations.append(a)

        return a,activations,zs

    def prediction (self,I):
        return self.softmax(self.forward_propagation(I)[0])

    def back_propagation (self,X,Y):
        m=X.shape[0]
        activations, zs = self.forward_propagation(X)[1], self.forward_propagation(X)[2]

        #Matrice des gradients, matrice mm forme mais avec zeros
        d_weights = [np.zeros_like(w) for w in self.weights]
        d_biases = [np.zeros_like(b) for b in self.biases]

        #On transforme une valeur catégorielle en vecteur binaire
        Y_one_hot = np.zeros((m, self.couches_sizes[-1]))
        Y_one_hot[np.arange(m), Y] = 1


couches_sizes = [784,150,250,10]  # liste de dim par couche
learning_rate = 0.1

P = MLP (couches_sizes, learning_rate)

test = [(X_test[i], Y_test[i]) for i in range(10000)]
I = test[0][0]
prediction = P.prediction(I)
print(prediction)