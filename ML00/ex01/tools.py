import numpy as np

def add_intercept(x):
	"""Adds a column of 1’s to the non-empty numpy.array x. Args:
	  x: has to be a numpy.array of dimension m * n.
	Returns:
	  X, a numpy.array of dimension m * (n + 1).
	  None if x is not a numpy.array.
	  None if x is an empty numpy.array.
	Raises:
	  This function should not raise any Exception.
	"""
	if not isinstance(x, np.ndarray) or len(x) == 0:
		print("Error x is not a numpy.array, or x is empty")
		return None
	if len(x.shape) == 1:
		x = x.reshape(-1, 1)
	colums = np.ones((x.shape[0], 1))
	x = np.concatenate((colums,x), axis=1)
	print(x)

if __name__=="__main__":
	x = np.arange(1,6)
	add_intercept(x)
	y = np.arange(1,10).reshape((3,3))
	add_intercept(y)