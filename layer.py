import activations
import numpy as np

class Layer:
	""" Implements a fully connected layer """

	def __init__(self, input_dims, output_dims, activation='sigmoid'):

		np.random.seed(10)

		self.input_dims = input_dims
		self.output_dims = output_dims

		self.activation = activations.Activation(activation)

		self.W = np.random.normal(0,0.2,(output_dims, input_dims))
		self.b = np.zeros((output_dims,1))
		self.dW = 0
		self.db = 0

		self.A 	= 0
		self.dA = 0

		self.A_prev = 0
		self.dA_prev = 0

		self.Z 	= 0
		self.dZ = 0


	def forward(self, A_prev):
		"""
		Implements the forward propagation

		Arguements:
		A_prev -- Activations from previous layer

		Returns:
		A -- Activations from this layer
		"""

		self.A_prev = A_prev
		self.Z = np.dot(self.W, self.A_prev) + self.b
		assert(self.Z.shape == (self.W.shape[0], self.A_prev.shape[1]))

		self.A = self.activation(self.Z)

		assert((self.A).shape == ((self.W).shape[0], (self.A_prev).shape[1]))

		return self.A


	def backward(self, dA):
		"""
		Implements the backward propagation

		Arguements:
		dA -- Post activation gradient for current layer

		Returns:
		dA_prev -- Gradient of the activation of the previous layer
		"""
		m = self.A_prev.shape[1]
		self.dZ = self.activation(self.Z,dA=dA,grad=True)
		self.dW = (1. / m) * np.dot(self.dZ, self.A_prev.T)
		self.db = (1. / m) * np.sum(self.dZ, axis=1, keepdims=True)
		self.dA_prev = np.dot(self.dW.T, self.dZ)

		assert(self.dA_prev.shape == self.A_prev.shape)
		assert(self.dW.shape == self.W.shape)
		assert(self.db.shape == self.b.shape)

		return self.dA_prev

	def update(self, lr):
		"""
		Updates the parameters of the layer

		Arguements:
		lr: learning rate
		"""
		self.W = self.W - lr*self.dW
		self.b = self.b - lr*self.db


