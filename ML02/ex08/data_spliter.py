import numpy as np

def data_spliter(x, y, proportion):
	"""
	Shuffles and splits the dataset (given by x and y) into a training and a test set,
	while respecting the given proportion of examples to be kept in the training set.
	Args:
	x: has to be an numpy.array, a matrix of dimension m * n.
	y: has to be an numpy.array, a vector of dimension m * 1.
	proportion: has to be a float, the proportion of the dataset that will be assigned to the
	training set.
	Return:
	(x_train, x_test, y_train, y_test) as a tuple of numpy.array
	None if x or y is an empty numpy.array.
	None if x and y do not share compatible dimensions.
	None if x, y or proportion is not of expected type.
	Raises:
	This function should not raise any Exception.
	"""
	if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(proportion, float)) or\
		x.shape[0] != y.shape[0] or np.size(x) == 0 or np.size(y) == 0 or proportion > 1 or proportion <= 0:
		print('Error in arguments')
		return None
	new_size = int(x.shape[0] * proportion)
	concatenated = np.concatenate((x, y), 1)

	np.random.shuffle(concatenated)
	x, y = concatenated[..., :-1], concatenated[..., -1:]
	x_train, x_test, y_train, y_test = x[:new_size], x[new_size:], y[:new_size], y[new_size:]
	return (x_train, x_test, y_train, y_test)

if __name__=="__main__":
	x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
	y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))

	print(data_spliter(x1, y, 0.8), end='\n\n')
	print(data_spliter(x1, y, 0.5), end='\n\n')

	x2 = np.array([[1, 42],
			   [300, 10],
			   [59,  1],
			   [300, 59],
			   [10, 42]])

	print(data_spliter(x2, y, 0.8), end='\n\n')
	print(data_spliter(x2, y, 0.5), end='\n\n')
