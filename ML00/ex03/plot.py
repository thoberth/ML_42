import matplotlib.pyplot as plt
import numpy as np

def plot(x, y, theta):
	"""Plot the data and prediction line from three non-empty numpy.array. Args:
	  x: has to be an numpy.array, a vector of dimension m * 1.
	  y: has to be an numpy.array, a vector of dimension m * 1.
	  theta: has to be an numpy.array, a vector of dimension 2 * 1.
	Returns:
		Nothing.
	Raises:
	  This function should not raise any Exceptions.
	"""
	if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(theta, np.ndarray))\
		or x.shape != (x.size,) or y.shape != (y.size,) or theta.shape != (2, 1):
		print("Error in arguments")
		return None
	def h(x): return theta[0] + theta[1] * x
	fig = plt.figure()
	plt.scatter(x, y)
	plt.plot(x, h(x), c='orange')
	plt.show()

if __name__=="__main__":
	x = np.arange(1, 6)
	y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
	theta1 = np.array([[4.5],[-0.2]])
	plot(x, y, theta1)
	theta2 = np.array([[-1.5],[2]])
	plot(x, y, theta2)
	theta3 = np.array([[3],[0.3]])
	plot(x, y, theta3)