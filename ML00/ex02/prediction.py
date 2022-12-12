import numpy as np

def predict_(x, theta):
	"""Computes the vector of prediction y_hat from two non-empty numpy.array. Args:
	  x: has to be an numpy.array, a vector of dimension m * 1.
	  theta: has to be an numpy.array, a vector of dimension 2 * 1.
	Returns:
	  y_hat as a numpy.array, a vector of dimension m * 1.
	  None if x and/or theta are not numpy.array.
	  None if x or theta are empty numpy.array.
	  None if x or theta dimensions are not appropriate.
	Raises:
	  This function should not raise any Exceptions.
	"""
	if not (isinstance(x, np.ndarray) and isinstance(theta, np.ndarray)) or theta.shape not in [(2,1), (2,)] or x.shape not in [(x.size,1), (x.size,)]:
		print("Error in argument")
		return None
	y_hat = theta[0] + theta[1] * x
	return y_hat

if __name__=="__main__":
	x = np.arange(1, 6)
	theta1 = np.array([[5], [0]])
	print(predict_(x, theta1))
	theta2 = np.array([[0], [1]])
	print(predict_(x, theta2))
	theta3 = np.array([[5], [3]])
	print(predict_(x, theta3))
	theta4 = np.array([[-3], [1]])
	print(predict_(x, theta4))
