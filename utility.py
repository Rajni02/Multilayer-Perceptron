import numpy as np

#utility function for one hot encoding
def to_onehot(Y, n_class):

	y = Y.reshape(-1,)
	m = y.shape[0]
	Y_one_hot = np.zeros((n_class,m))
	for i in range(m):
		Y_one_hot[y[i]][i] = 1
	return Y_one_hot


