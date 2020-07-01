import numpy as np
import layer

class Model:
	""" Creates a Neural Network Model """

	def __init__(self, layer_list, loss='mse'):
		self.layer_list = layer_list
		self.loss = loss			#loss for the model
		self.out = 0				#final layer ouptut

	def model_forward(self, X):
		"""
		Single forward propagation through the entire model

		Arguements:
		X -- training data
		"""
		A = X
		for l in self.layer_list:
			A_prev = A
			A = l.forward(A_prev)

		self.out = self.layer_list[-1].A
		

	def model_backward(self, Y):
		"""
		Single backward propagation through the entire model
		
		Arguements:
		Y -- labels
		"""
		AL = self.out
		if self.loss == 'cross-entropy' :
			dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  #categorical cross entropy loss gradient
		elif self.loss == 'mse' :
			dAL = (AL - Y)										   #mse loss gradient

		dA_prev = dAL
		for l in reversed(self.layer_list):
			dA_prev_temp = dA_prev
			dA_prev = l.backward(dA_prev_temp)			

	def model_update(self, lr):
		"""
		Updates the parameters of the entire model

		Arguements:
		lr -- Learning rate
		"""

		for l in self.layer_list:
			l.update(lr)

	def compute_cost(self, Y):       

		m = Y.shape[1]
		cost = (1. / m)*np.sum(np.square(self.out - Y))

		return cost

	def train(self, X_train, Y_train, epochs=2000,lr=0.1):

		epoch_table = []
		for i in range(1,epochs+1):
			self.model_forward(X_train)
			cost = self.compute_cost(Y_train)
			if i % 100 == 0 : print("{}th Epoch Cost is -->{}".format(i,cost))
			epoch_table.append(cost)
			self.model_backward(Y_train)
			self.model_update(lr)

		return epoch_table


	def evaluate(self, X_test, Y_test):

		self.model_forward(X_test)
		Y_predicted = self.out

		m = Y_test.shape[1]
		n_class = Y_test.shape[0]


		Y_predicted_class =  np.argmax(Y_predicted, axis=0)
		Y_actual_class = np.argmax(Y_test, axis=0)

		conf_matrix = np.zeros((n_class,n_class))

		for i in range(m):
			predicted_class = Y_predicted_class[i]
			actual_class = Y_actual_class[i]
			conf_matrix[predicted_class][actual_class] += 1

		num_correct_pred = 0
		for i in range(n_class):
			num_correct_pred += conf_matrix[i][i]

		accuracy = num_correct_pred / m
		return conf_matrix, accuracy







		
