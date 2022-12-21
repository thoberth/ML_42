from vector import Vector

class Matrix:
	def __init__(self, args):
		if (isinstance(args, list) and not all(isinstance(elem, list) for elem in args))\
			or (isinstance(args, tuple) and len(args) != 2) or\
			not (isinstance(args, list) and isinstance(args, tuple)):
			print('Error Matrix must be initiallized with tuple or list of list of float')
			return None
		if isinstance(args, list):
			test_len = len(args[0])
			for elem in args[1:]:
				if len(elem) != test_len:
					print('Error Matrix have rows of same lenght')
					return None
			self.data = args
			self.shape = (len(args), test_len)
		else:
			self.shape = args
			self.data = [[0. * self.shape[1]] * self.shape[1]]
			# for i in range(self.shape[0]):
			# 	for _ in range(self,shape[1]):


	def T(self): pass

	# add : only matrices of same dimensions.
	def __add__(self, other):
		if other.shape != self.shape:
			print("Error Matrices should have the same dimensions")
			return None

	def __radd__(self, other):
		if other.shape != self.shape:
			print("Error Matrices should have the same dimensions")
			return None

	# sub : only matrices of same dimensions.
	def __sub__(self, other):
		if other.shape != self.shape:
			print("Error Matrices should have the same dimensions")
			return None

	def __rsub__(self, other):
		if other.shape != self.shape:
			print("Error Matrices should have the same dimensions")
			return None

	# div : only scalars.
	def __truediv__(self, scalar):
		if not (isinstance(scalar, int) or isinstance(scalar, float)):
			print("only scalar for division")
			return None

	def __rtruediv__(self, scalar):
		if not (isinstance(scalar, int) or isinstance(scalar, float)):
			print("only scalar for division")
			return None

	# mul : scalars, vectors and matrices , can have errors with vectors and matrices, # returns a Vector if we perform Matrix * Vector mutliplication.
	def __mul__(self, other): pass
	def __rmul__(self, other): pass


	def __str__(self, other):
		return ('Matrix({})'.format(self.data))

	def __repr__(self, other):
		return ('Matrix of shape {},\nvalue: {}'.format(self.shape, self.data))

if __name__ == "__main__":
	pass