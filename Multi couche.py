import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import random

t = time.time()
#train_data = pd.read_csv('C:/Users/Utilisateur/Documents/L3 TSE/S5/Projet informatique magistère/mnist_train.csv')
#test_data = pd.read_csv("C:/Users/Utilisateur/Documents/L3 TSE/S5/Projet informatique magistère/mnist_test.csv")
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
    def __init__(self, taille_couches, tx_app, activation_function, activation_derivative):

        self.taille_couches = taille_couches  # liste de dim par couche
        self.tx_app = tx_app
        self.L = len(taille_couches)  # Nombre de ciouches
        self.poids = [] #liste de matrices
        self.biais = [] #liste de vecters
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

        for l in range(self.L-1):
            input_sz = taille_couches[l]
            output_sz = taille_couches[l+1]
            self.poids.append(np.random.randn(input_sz, output_sz) * 0.01)
            self.biais.append((np.random.randn(output_sz) * 0.01).reshape(1, -1))




    def softmax(self, v):
        if v.ndim == 1:
            v = v.reshape(1, -1)
        return np.exp(v) / np.sum(np.exp(v), axis=1, keepdims=True)

    def forward_propagation(self, A):
        activations = [] #stocke les activations des couches
        zs = []  #stocke les valeurs avant activation
        A0 = self.activation_function(A)
        activations.append(A0)

        for l in range(self.L-1):
            v = np.dot(A, self.poids[l]) + self.biais[l]
            zs.append(v) # besoin de stocker pour backward
            A = self.activation_function(v)
            activations.append(A)

        return A, activations, zs
        #I represente la sortie finale du reseau(avant softmax), vecteur de taille [1,10]
        #activations liste avec toutes les activations des couches
        #zs liste avec toutes les valeurs avant activations

    def prediction (self,I):
        return self.softmax(self.forward_propagation(I)[0])

    def back_propagation(self, X, Y):
        m = X.shape[0]  # nombre d'exemples, ici c'est 1 image donc m=1

        activations, zs = self.forward_propagation(X)[1], self.forward_propagation(X)[2]

        err_poids = [np.zeros((1,i)) for i in self.taille_couches[1:]]
        err_biais = [np.zeros((1,i)) for i in self.taille_couches[1:]]

        Y_one_hot = np.zeros((1, self.taille_couches[-1]))
        Y_one_hot[0,Y] = 1

        err_poids[-1] = activations[-1] - Y_one_hot
        err_biais[-1] = err_poids[-1]

        for l in range(self.L-3,-1,-1): # de la dernière couche cachée à la première
            # calcul erre pour chaque couche cachée
            err_poids[l] = np.dot(err_poids[l + 1], self.poids[l+1].T) * self.activation_derivative(activations[l+1])
            err_biais[l] = np.mean(err_poids[l], axis=0).reshape(1, -1)

        for l in range(len(self.poids)):  # maj que L-1 couches
            self.poids[l] -= self.tx_app * np.dot(activations[l].T, err_poids[l])
            self.biais[l] -= self.tx_app * err_biais[l]



taille= [784,10]  # liste de dim par couche si trop ça perde en performance car surapprentissage
activations_fct = []
taux_app= 0.05

def relu_function(v):
    return np.maximum(0, v)
def relu_derivative(a):
    return (a > 0).astype(float)
activations_fct.append((relu_function,relu_derivative))

def sigmoide_function(v):
    return 1 / (1 + np.exp(-v))
def sigmoide_derivative(a):
    return a * (1 - a)
activations_fct.append((sigmoide_function,sigmoide_derivative))

def selu_activation(v, alpha=1.6733, lambd=1.0507):
    return lambd * np.where(v > 0, v, alpha * (np.exp(v) - 1))
def selu_derivative(v, alpha=1.6733, lambd=1.0507):
    return lambd * np.where(v > 0, 1, alpha * np.exp(v))
activations_fct.append((selu_activation,selu_derivative))

resultats = {}


for nb_couches in range(1, 5):
    taille.insert(-1, random.randint(10, 120))
    print(f"Itération {nb_couches}: {taille}")

    for fct in activations_fct:
        P = MLP(taille, taux_app, fct[0], fct[1])

        for i in range(len(X_train)):
            X_image = X_train[i].reshape(1, -1)
            Y_image = np.array([Y_train[i]])
            P.back_propagation(X_image, Y_image)

        r, br = 0, 0
        for i in range(len(X_test)):
            r += 1
            X_image = X_test[i].reshape(1, -1)
            Y_image = Y_test[i]
            prediction = np.argmax(P.prediction(X_image))

            if prediction == Y_image:
                br += 1

        performance = (br / r) * 100  # Taux de bonnes réponses en pourcentage
        print(f"Taux de bonnes réponses pour {fct[0]} : {performance:.2f}%")

        # Stocker les performances
        if fct[0] not in resultats:
            resultats[fct[0]] = []
        resultats[fct[0]].append((nb_couches, performance))


plt.figure(figsize=(8, 6))

for fct, data in resultats.items():
    couches, performances = zip(*data)
    plt.plot(couches, performances, marker='o', label=fct)

plt.xlabel("Nombre de couches cachées")
plt.ylabel("Taux de bonnes réponses (%)")
plt.title("Performance des fonctions d'activation en fonction du nombre de couches")
plt.legend()
plt.grid(True)
plt.show()

"""
for taux_app in np.arange(0.01,1,0.01) :
    P = MLP(taille, taux_app)
    t = time.time()
    # Entraînement
    for rep in range (1) :
        for i in range(len(X_train)):
                X_image= X_train[i].reshape(1, -1)  #forme (1, 784)
                Y_image = np.array([Y_train[i]])  #forme (1,)
                P.back_propagation(X_image, Y_image)  #mise à jour poids avec backpropagation
    print(time.time() - t)


    # Test
    r = 0
    br = 0
    for i in range(len(X_test)):
        r += 1
        X_image = X_test[i].reshape(1, -1)  # forme (1, 784)
        Y_image = Y_test[i]
        prediction = np.argmax(P.prediction(X_image))
        #print(Y_image,prediction)  # forme (1,)

        if prediction == Y_image:
            br += 1

    print(f"Taux de bonnes réponses : {taux_app  },{br / r * 100:.2f}%")
"""