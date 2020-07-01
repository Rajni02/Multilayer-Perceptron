import numpy as np
class Activation:
	""" implements all activation function """	

	def __init__(self, activation):

		self.activation_list = ['signum','sigmoid','softmax','relu','tanh']
		self.activation = activation

		if self.activation not in self.activation_list:
			raise ValueError('{} function not defined '.format(activation))

	

	def __call__(self, Z, dA=0, grad=False):

		if self.activation == 'signum':
			if not grad: return signum(Z)
			else:
				raise ValueError("Signum is not differentiable") 

		elif self.activation == 'sigmoid':
			if not grad: return sigmoid(Z)
			else: return sigmoid_backward(dA, Z)

		elif self.activation == 'relu':
			if not grad: return relu(Z)
			else: return relu_backward(dA, Z)

		elif self.activation == 'softmax':
			if not grad: return softmax(Z)
			else: return softmax_backward(dA, Z)
		
		


def signum(x):
	if x > 0.0 : return 1
	else : return 0

def sigmoid(Z):
	"""
	Implemets the sigmoid activation function

	Arguements:
	Z -- numpy array

	Returns:
	A -- output of sigmoid(Z), same shape as Z	
	"""

	A = 1/(1 + np.exp(-Z))

	return A
	
def sigmoid_backward(dA, Z):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:	
    dA -- post-activation gradient
    Z  -- affine output    

    Returns:
    dZ -- Gradient of the cost with respect to Z    
    """
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ



def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer

    Returns:
    A -- Post-activation output 
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)    
     
    return A

def relu_backward(dA, Z):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    Z  -- affine output

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """ 
    
    dZ = np.array(dA, copy=True) 
    
    # When z <= 0, setting dz to 0 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def softmax(Z):
	A = np.exp(Z) / np.sum(np.exp(Z), axis = 0)	
	return A

def softmax_backward(dA, Z):
	A = softmax(Z)
	return A*dA
   
  
