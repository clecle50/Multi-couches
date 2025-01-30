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

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        np.random.seed(42)
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(hidden_size, input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size)
        self.b2 = np.zeros((output_size, 1))

    def forward(self, X):
        self.Z1 = np.dot(self.W1, X.T) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2.T

    def train(self, X, Y, rep=10000):
        for rep in range(rep):
            self.forward(X)  # Forward pass
            self.backward(X, Y)  # Backpropagation

            # Calcul de la perte (MSE)
            loss = np.mean((Y - self.A2.T) ** 2)




