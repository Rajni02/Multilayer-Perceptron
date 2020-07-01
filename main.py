import numpy as np
from layer import Layer
from model import Model
from utility import *
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()

X = iris.data.T
Y = iris.target

#normalizing the dataset
x_mean = np.mean(X, axis=1).reshape(-1,1)
x_std = np.std(X, axis=1).reshape(-1,1)

X = X - x_mean
X = X / x_std

#train -- test split
X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y, test_size=0.3, random_state=100)

X_train = X_train.T
X_test = X_test.T

#one hot repr. of labels
Y_train = to_onehot(Y_train,3)
Y_test = to_onehot(Y_test,3)


#defining the layers for the model
L1 = Layer(4,8,activation='relu') 				#1st hidden layer
L2 = Layer(8,3,activation='sigmoid')			#final output layer

L = [L1,L2]										#L: list of layers for the sequential model

m = Model(L,loss='mse')							#creating the model, with the list of layers

history = m.train(X_train,Y_train,lr=0.1,epochs=800)	#training the model


conf_matrix_test, accuracy_test   = m.evaluate(X_test, Y_test)
conf_matrix_train, accuracy_train = m.evaluate(X_train, Y_train)


print("Training Data Confusion matrix: ")
print(conf_matrix_train)
print("Train Accuracy = {}\n".format(accuracy_train))

print("Test Data Confusion matrix: ")
print(conf_matrix_test)
print("Test Accuracy = {}".format(accuracy_test))


plt.plot(history)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()

#------------------------------------------------
'''
digits = datasets.load_digits()
X = digits.data
Y = digits.target
class_names = digits.target_names
no_of_classes = class_names.shape[0]

# Normalizing the dataset
x_mean = np.mean(X, axis=0)
x_std = np.std(X, axis=0) + 0.001
X = X - x_mean
X = X / x_std

# Train_test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.7, random_state=100)

X_train = X_train.T
X_test = X_test.T

# One Hot encoding the lebel
Y_train = to_onehot(Y_train,no_of_classes)
Y_test = to_onehot(Y_test,no_of_classes)


L1 = Layer(64,100,activation='relu')

L2 = Layer(100,10,activation='sigmoid')


l = [L1,L2]

m = Model(l)
history = m.train(X_train,Y_train,lr=0.5, epochs=2000)


conf_matrix_test, accuracy_test   = m.evaluate(X_test, Y_test)
conf_matrix_train, accuracy_train = m.evaluate(X_train, Y_train)


print("Training Data Confusion matrix: ")
print(conf_matrix_train)
print("Train Accuracy = {}\n".format(accuracy_train*100))

print("Test Data Confusion matrix: ")
print(conf_matrix_test)
print("Test Accuracy = {}".format(accuracy_test*100))


plt.plot(history)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()'''
