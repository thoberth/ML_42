import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath("../ex05/"))
from other_losses import *

def plot_with_loss(x, y, theta):
	"""Plot the data and prediction line from three non-empty numpy.ndarray.
	Args:
	x: has to be an numpy.ndarray, a vector of dimension m * 1.
	y: has to be an numpy.ndarray, a vector of dimension m * 1.
	theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
	Returns:
	Nothing.
	Raises:
	This function should not raise any Exception.
	"""
	if not (isinstance(y, np.ndarray) and isinstance(x, np.ndarray) and isinstance(theta, np.ndarray)) or\
		y.shape not in [(y.size,), (y.size, 1)] or x.shape not in [(x.size,), (x.size, 1)]\
			or theta.shape not in [(2,), (2, 1)]:
		print("Error arguments are wrong")
		return None
	def h(x): return theta[0] + theta[1] * x
	predict = h(x)
	mse = mse_(y, predict)
	fig = plt.figure()
	plt.plot(x, predict, label='prediction', c='orange') #ligne
	for i in range(len(predict)):
		plt.stem(x[i], y[i], linefmt='r--', bottom=predict[i])
	plt.title('Cost : {}'.format(mse))
	plt.grid()
	plt.legend()
	plt.show()

if __name__ == "__main__":
	x = np.arange(1, 6)
	y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])

	theta1= np.array([18,-1])
	plot_with_loss(x, y, theta1)

	theta2 = np.array([14, 0])
	plot_with_loss(x, y, theta2)

	theta3 = np.array([12, 0.8])
	plot_with_loss(x, y, theta3)
