import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / 1 - np.exp(-x)

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


class Multi():

    def __init__(self, train, chiffre, seuil, teta):
        self.chiffre = chiffre
        self.seuil = seuil
        self.base = train
        self.teta = teta
        self.taille_image = len(self.base[0][0].flatten())
        self.w = np.zeros(self.taille_image + 1)
        self.input_size =
        self.output_size =



