import os, sys
import numpy as np
from sklearn.metrics import confusion_matrix
sys.path.append(os.path.abspath("../ex06/"))
from other_metrics import count_metrics

def confusion_matrix_(y_true, y_hat, labels=None):
	"""
	Compute confusion matrix to evaluate the accuracy of a classification.
	Args:
		y_true: numpy.ndarray for the correct labels
		y_hat: numpy.ndarray for the predicted labels
		labels: Optional, a list of labels to index the matrix.
				This may be used to reorder or select a subset of labels. (default=None)
	Returns:
		The confusion matrix as a numpy ndarray.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	if labels == None:
		labels = np.unique((y_true, y_hat)).tolist()
	matrix = np.zeros((len(labels), len(labels)), dtype = 'int')
	for true, hat, in zip(y_true, y_hat):
		if true in labels and hat in labels:
			matrix[labels.index(true)][labels.index(hat)] += 1
	return matrix


if __name__ == "__main__":
	y_hat = np.array([['norminet'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['bird']])
	y = np.array([['dog'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['norminet']])
	# Example 1:
	# your implementation
	print(confusion_matrix_(y, y_hat))
	# sklearn implementation
	print(confusion_matrix(y, y_hat))
	# Example 2:
	# your implementation
	print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']))
	# sklearn implementation
	print(confusion_matrix(y, y_hat, labels=['dog', 'norminet']))