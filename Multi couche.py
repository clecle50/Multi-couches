import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt



t = time.time()
#train_data = pd.read_csv('C:/Users/Utilisateur/Documents/L3 TSE/Projet informatique magistère/mnist_train.csv')
#test_data = pd.read_csv("C:/Users/Utilisateur/Documents/L3 TSE/Projet informatique magistère/mnist_test.csv")
train_data = pd.read_csv('C:/Users/ghani/Documents/Desktop/TSE/Licence 3/S1/Projet Magistere/mnist_train.csv')
test_data = pd.read_csv('C:/Users/ghani/Documents/Desktop/TSE/Licence 3/S1/Projet Magistere/mnist_test.csv')

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
        """
        Initialise le MLP avec un nombre variable de couches et de neurones.

        :param layer_sizes: Liste contenant le nombre de neurones par couche (entrée + cachées + sortie).
        :param learning_rate: Taux d'apprentissage pour la mise à jour des poids.
        """

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

    def forward_propagation(self, M):
        a = M
        for l in range(self.L):
            v = np.dot(a, self.weights[l]) + self.biases[l]
            a = self.activation_function(v)
        return a

    def activation_function(self, v):
        return 1 / (1 + np.exp(-v))

    def softmax(self, v):
        # Ensure v is a 2D array
        if v.ndim == 1:
            v = v.reshape(1, -1)
        return np.exp(v) / np.sum(np.exp(v), axis=1, keepdims=True)

    def prediction (self,I):
        return self.softmax(self.forward_propagation(I))



couches_sizes = [784,150,250,10]  # liste de dim par couche
learning_rate = 0.1

P = MLP (couches_sizes, learning_rate)

test = [(X_test[i], Y_test[i]) for i in range(10000)]
I = test[0][0]
prediction = P.prediction(I)
print(prediction)