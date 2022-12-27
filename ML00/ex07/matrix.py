from vector import Vector

class Matrix:
	def __init__(self, args):
		if (isinstance(args, list) and not all(isinstance(elem, list) for elem in args))\
		or not (isinstance(args, list) or isinstance(args, tuple)):
			raise Exception('Error Matrix must be initiallized with tuple of size 2\
or list of list of float')
		if isinstance(args, list):
			test_len = len(args[0])
			for elem in args[1:]:
				if len(elem) != test_len:
					raise Exception('Error Matrix have rows of same lenght')
			self.data = args
			self.shape = (len(args), test_len)
		else:
			self.shape = args
			self.data = [[0. for _ in  range(self.shape[1])] for _ in range(self.shape[0])]

	def T(self): # transpose a matrice
		new_matrix = Matrix((self.shape[1], self.shape[0]))
		for x in range(len(new_matrix.data)):
			for y in range(len(new_matrix.data[x])):
				new_matrix.data[x][y] = self.data[y][x]
		self.data = new_matrix.data
		self.shape = new_matrix.shape

	# add : only matrices of same dimensions.
	def __add__(self, other):
		if not isinstance(other, Matrix) or other.shape != self.shape:
			raise Exception("Error: Only matrices of same dimensions could be add up")
		new_matrix = Matrix(self.shape)
		for x in range(len(self.data)):
			for y in range(len(self.data[x])):
				new_matrix.data[x][y] = self.data[x][y] + other.data[x][y]
		return new_matrix

	def __radd__(self, other):
		if not isinstance(other, Matrix) or other.shape != self.shape:
			raise Exception("Error: Only matrices of same dimensions could be add up")
		new_matrix = Matrix(self.shape)
		for x in range(len(self.data)):
			for y in range(len(self.data[x])):
				new_matrix.data[x][y] = self.data[x][y] + other.data[x][y]
		return new_matrix

	# sub : only matrices of same dimensions.
	def __sub__(self, other):
		if not isinstance(other, Matrix) or other.shape != self.shape:
			raise Exception("Error: Only matrices of same dimensions could be substracted")
		new_matrix = Matrix(self.shape)
		for x in range(len(self.data)):
			for y in range(len(self.data[x])):
				new_matrix.data[x][y] = self.data[x][y] - other.data[x][y]
		return new_matrix

	def __rsub__(self, other):
		if not isinstance(other, Matrix) or other.shape != self.shape:
			raise Exception(
				"Error: Only matrices of same dimensions could be substracted")
		new_matrix = Matrix(self.shape)
		for x in range(len(self.data)):
			for y in range(len(self.data[x])):
				new_matrix.data[x][y] = other.data[x][y] - self.data[x][y]
		return new_matrix

	# div : only scalars.
	def __truediv__(self, scalar):
		if not (isinstance(scalar, int) or isinstance(scalar, float)):
			raise Exception("only scalar for division")
		if scalar == 0:
			raise Exception('DivisionByZero')
		new_matrix = Matrix(self.shape)
		for x in range(len(self.data)):
			for y in range(len(self.data[x])):
				new_matrix.data[x][y] = self.data[x][y] / scalar
		return new_matrix

	def __rtruediv__(self, scalar):
		if not (isinstance(scalar, int) or isinstance(scalar, float)):
			raise Exception("only scalar for division")
		if scalar == 0:
			raise Exception('DivisionByZero')
		new_matrix = Matrix(self.shape)
		for x in range(len(self.data)):
			for y in range(len(self.data[x])):
				new_matrix.data[x][y] = scalar / self.data[x][y]
		return new_matrix

	# mul : scalars, vectors and matrices , can have errors with vectors and matrices,
	# returns a Vector if we perform Matrix * Vector mutliplication.
	def __mul__(self, other):
		new_matrix = Matrix(self.shape)
		# same shape product
		if isinstance(other, Matrix) and self.shape == other.shape:
			for x in range(len(self.data)):
				for y in range(len(self.data[x])):
					new_matrix.data[x][y] = self.data[x][y] * other.data[x][y]
		# different shape product
		elif isinstance(other, Matrix) and self.shape[1] == other.shape[0]:
			new_matrix = Matrix((self.shape[0], other.shape[1]))
			for x in range(len(new_matrix.data)):
				for y in range(len(new_matrix.data[x])):
					for i in range(other.shape[0]):
						new_matrix.data[x][y] += self.data[x][i] * other.data[i][y]
		# matrice * vector product
		elif isinstance(other, Vector) and self.shape[1] == other.shape[0]:
			new_matrix = self.__mul__(Matrix(other.values))
			new_matrix = Vector(new_matrix.data)
		# matrice * scalar product 
		elif isinstance(other, int) or isinstance(other, float):
			for x in range(len(self.data)):
				for y in range(len(self.data[x])):
					new_matrix.data[x][y] = self.data[x][y] * other
		else:
			raise Exception('Error, Matrices should be multiplied by Matrices\
(with compatible shapes), Vectors or scalar')
		return new_matrix

	def __rmul__(self, other):
		return self.__mul__(other)

	def __is_squared(matrice):
		if matrice.shape[0] == matrice.shape[1]:
			return True
		return False

	def __print_value(self):
		to_print = "["
		for x in range(len(self.data)):
			to_print += str(self.data[x])
			if x != (len(self.data) - 1):
				to_print += '\n '
		return(to_print + ']')

	def __str__(self):
		return ('{}'.format(self.__print_value()))

	def __repr__(self):
		return ('Matrix of shape {},\nvalue: {}'.format(self.shape, self.__print_value()))
