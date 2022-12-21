# Column vector of shape (n, 1)
# print(Vector([[0.0], [1.0], [2.0], [3.0]]).shape)
# Expected output
# (4,1)

# print(Vector([[0.0], [1.0], [2.0], [3.0]]).values)
# Expected output
# [[0.0], [1.0], [2.0], [3.0]]

# Row vector of shape (1, n)
# print(Vector([[0.0, 1.0, 2.0, 3.0]]).shape)
# Expected output
# (1,4)

# print(Vector([[0.0, 1.0, 2.0, 3.0]]).values)
# Expected output
# [[0.0, 1.0, 2.0, 3.0]]

# Example 1:
# v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
# print(v1.shape)
# Expected output:
# (4,1)

# print(v1.T())
# Expected output:
# Vector([[0.0, 1.0, 2.0, 3.0]])

# print(v1.T().shape)
# Expected output:
# (1,4)

# Example 2:
# v2 = Vector([[0.0, 1.0, 2.0, 3.0]])
# print(v2.shape)
# Expected output:
# (1,4)

# print(v2.T())
# Expected output:
# Vector([[0.0], [1.0], [2.0], [3.0]])

# print(v2.T().shape)
# Expected output:
# (4,1)

# Example 1:
# v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
# v2 = Vector([[2.0], [1.5], [2.25], [4.0]])
# print(v1.dot(v2))
# Expected output:
# 18.0

# v3 = Vector([[1.0, 3.0]])
# v4 = Vector([[2.0, 4.0]])
# print(v3.dot(v4)) = (1 x 2) + (3 x 4) = 14.0
# Expected output:
# 14.0

# En algèbre linéaire, les nombres réels qui multiplient les vecteurs dans un espace vectoriel sont appelés des scalaires.
# Cette multiplication par un scalaire, qui permet de multiplier un vecteur par un nombre pour produire un vecteur,
# correspond à la loi externe de l'espace vectoriel.

# Plus généralement, dans un K-espace vectoriel, les scalaires sont les éléments de K,
# où K peut être l'ensemble des nombres complexes ou n'importe quel autre corps.

# D'autre part, un produit scalaire (à ne pas confondre avec la multiplication par un scalaire) peut être défini sur un espace vectoriel,
# permettant à deux vecteurs d'être multipliés entre eux pour donner un scalaire.
# Un espace vectoriel de dimension finie et muni d'un produit scalaire est appelé un espace vectoriel euclidien.
from typing import List, Tuple

t_Vector = List[List[float]]

class Vector:
	"""class Vector: if shape ==\n
	(1, n) vector is a row\n
	(n, 1) vector is a Column\n
	(n, n) this is not anymore a vector this is a Matrix\n"""

	def __init__(self, *args): # By default, the vectors are generated as classical column vectors if initialized with a size or range.
		if (isinstance(args[0], list) and all(isinstance(i, list) for i in args[0]) and args.__len__() == 1): # Vector(list(list(float)))
			self.values = args[0]
			if (self.values.__len__() == 1):
				self.shape = tuple([1, self.values[0].__len__()])
			else:
				self.shape = tuple([self.values.__len__(), 1])
		elif (isinstance(args[0], int) and args.__len__() == 1): # Vector(size)
			size = args[0]
			if (size < 0):
				print("Error: size is negative, so size is 1 by default")
				size = 1
			new_list = [[0]]
			for i in range(1, size):
				new_list.append([i])
			self.values = new_list
			self.shape = tuple([self.values.__len__(), 1])
		elif (isinstance(args[0], int) and isinstance(args[1], int) and args.__len__() == 2): # Vector(a, b)
			a = args[0]
			b = args[1]
			if (a > b):
				print("Error: {} is superior than {}, value are Vector([[{}]])".format(a, b, a))
				b = a
			new_list = [[a]]
			for i in range(a + 1, b):
				new_list.append([i])
			print(new_list)
			self.values = new_list
			self.shape = tuple([self.values.__len__(), 1])

	def dot(self, other: "Vector"): # dot produce by an other vector of same shape
		result = 0.0
		if (self.shape[0] == 1 and other.shape[0] == 1):
			if (self.values[0].__len__() != other.values[0].__len__()):
				print('Error, Row Vector don\'t have the same dimension')
				return
			for i in range(self.values[0].__len__()):
				result += (self.values[0][i] * other.values[0][i])
		elif (self.shape[1] == 1 and other.shape[1] == 1):
			if (self.values.__len__() != other.values.__len__()):
				print('Error, Column Vector don\'t have the same dimension')
				return
			for i in range(self.values.__len__()):
				result += (self.values[i][0] * other.values[i][0])
		else:
			print('ERROR, Vector do not have the same shape')
			return
		return result

	def T(self):
		""" Row Vector to Column Vector or the opposite """
		new_list = [[self.values[0][0]]]
		if (self.shape[0] == 1):
			for i in range(1, self.values[0].__len__()):
				new_list.append([self.values[0][i]])
		else:
			for i in range(1, self.values.__len__()):
				new_list[0].append(self.values[i][0])
		return (Vector(new_list))

	def __str__(self):
		return ('Vector({})'.format(self.values))

	def __repr__(self):
		return ('Vector({})'.format(self.values))

	def __add__(self, other): # add & radd : only vectors of same shape.
		if (isinstance(other, Vector) != True):
			raise NotImplementedError('NotImplementedError: Sum of a scalar by a Vector is not defined here.')
		new_list = [[self.values[0][0] + other.values[0][0]]]
		if (self.shape[0] == 1 and other.shape[0] == 1):
			if (self.values[0].__len__() != other.values[0].__len__()):
				print('Error, Row Vector don\'t have the same dimension')
				return
			for i in range(1, self.values[0].__len__()):
					new_list.append([self.values[0][i] + other.values[0][i]])
		elif (self.shape[1] == 1 and other.shape[1] == 1):
			if (self.values.__len__() != other.values.__len__()):
				print('Error, Column Vector don\'t have the same dimension')
				return
			for i in range(1, self.values.__len__()):
				new_list[0].append(self.values[i][0] + other.values[i][0])
		else:
			print('ERROR, Vector do not have the same shape')
			return
		return (Vector(new_list))

	def __radd__(self, other):
		if (isinstance(other, Vector) != True):
			raise NotImplementedError('NotImplementedError: Sum of a scalar by a Vector is not defined here.')
		new_list = [[self.values[0][0] + other.values[0][0]]]
		if (self.shape[0] == 1 and other.shape[0] == 1):
			if (self.values[0].__len__() != other.values[0].__len__()):
				print('Error, Row Vector don\'t have the same dimension')
				return
			for i in range(1, self.values[0].__len__()):
					new_list.append([self.values[0][i] + other.values[0][i]])
		elif (self.shape[1] == 1 and other.shape[1] == 1):
			if (self.values.__len__() != other.values.__len__()):
				print('Error, Column Vector don\'t have the same dimension')
				return
			for i in range(1, self.values.__len__()):
				new_list[0].append(self.values[i][0] + other.values[i][0])
		else:
			print('ERROR, Vector do not have the same shape')
			return
		return (Vector(new_list))

	def __sub__(self, other): # sub & rsub: only vectors of same shape.
		if (isinstance(other, Vector) != True):
			raise NotImplementedError('NotImplementedError: Subtraction of a scalar by a Vector is not defined here.')
		new_list = [[self.values[0][0] - other.values[0][0]]]
		if (self.shape[0] == 1 and other.shape[0] == 1):
			if (self.values[0].__len__() != other.values[0].__len__()):
				print('Error, Row Vector don\'t have the same dimension')
				return
			for i in range(1, self.values[0].__len__()):
					new_list.append([self.values[0][i] - other.values[0][i]])
		elif (self.shape[1] == 1 and other.shape[1] == 1):
			if (self.values.__len__() != other.values.__len__()):
				print('Error, Column Vector don\'t have the same dimension')
				return
			for i in range(1, self.values.__len__()):
				new_list[0].append(self.values[i][0] - other.values[i][0])
		else:
			print('ERROR, Vector do not have the same shape')
			return
		return (Vector(new_list))

	def __rsub__(self, other):
		if (isinstance(other, Vector) != True):
			raise NotImplementedError('NotImplementedError: Subtraction of a scalar by a Vector is not defined here.')
		new_list = [[self.values[0][0] - other.values[0][0]]]
		if (self.shape[0] == 1 and other.shape[0] == 1):
			if (self.values[0].__len__() != other.values[0].__len__()):
				print('Error, Row Vector don\'t have the same dimension')
				return
			for i in range(1, self.values[0].__len__()):
					new_list.append([self.values[0][i] - other.values[0][i]])
		elif (self.shape[1] == 1 and other.shape[1] == 1):
			if (self.values.__len__() != other.values.__len__()):
				print('Error, Column Vector don\'t have the same dimension')
				return
			for i in range(1, self.values.__len__()):
				new_list[0].append(self.values[i][0] - other.values[i][0])
		else:
			print('ERROR, Vector do not have the same shape')
			return
		return (Vector(new_list))

	def __truediv__(self, other: float): # truediv : only with scalars (to perform division of Vector by a scalar).
		if (other == 0):
			raise ZeroDivisionError('ZeroDivisionError: division by zero.')
		print("before div:", self.values)
		result = [[self.values[0][0] / other]]
		if (self.shape[0] == 1):
			for i in range(1, self.values[0].__len__()):
				result[0].append(self.values[0][i] / other)
		else:
			for i in range(1, self.values.__len__()):
				result.append([self.values[i][0] / other])
		return (Vector(new_list))

	def __rtruediv__(self, other): # rtruediv : raises an NotImplementedError with the message "Division of a scalar by a Vector is not defined here."
		raise NotImplementedError('NotImplementedError: Division of a scalar by a Vector is not defined here.')

	def __mul__(self, other: float): # mul & rmul: only scalars (to perform multiplication of Vector by a scalar).
		print("before mul:", self.values)
		result = [[self.values[0][0] * other]]
		if (self.shape[0] == 1):
			for i in range(1, self.values[0].__len__()):
				result[0].append(self.values[0][i] * other)
		else:
			for i in range(1, self.values.__len__()):
				result.append([self.values[i][0] * other])
		return (Vector(new_list))

	def __rmul__(self, other : float):
		print("before mul:", self.values)
		result = [[self.values[0][0] * other]]
		if (self.shape[0] == 1):
			for i in range(1, self.values[0].__len__()):
				result[0].append(self.values[0][i] * other)
		else:
			for i in range(1, self.values.__len__()):
				result.append([self.values[i][0] * other])
		return (Vector(new_list))
