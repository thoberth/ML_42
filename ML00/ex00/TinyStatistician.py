import math
class TinyStatician:
	def __init__(self):
		pass

	def	mean(self, x): #moyenne
		if len(x) == 0:
			return None
		r = 0
		for i in x:
			r+=i
		r/=len(x)
		return float(r)

	def median(self, x):
		if len(x) == 0:
			return None
		x.sort()
		r = len(x)
		if r % 2 == 0:
			r = x[(int(r/2)) - 1]
		else :
			r = x[(int((r+1)/2)) - 1]
		return float(r)

	def quartiles(self, x):
		if len(x) == 0:
			return None
		x.sort()
		q1 = x[math.ceil((len(x)/4)) - 1]
		q3 = x[math.ceil((3*len(x)/4)) - 1]
		return [float(q1), float(q3)]

	def var(self, x):
		if len(x) == 0:
			return None
		x.sort()
		r = 0.0
		mean = self.mean(x)
		for i in x:
			r += math.pow(i - mean, 2)
		return(r / len(x))

	def std(self, x):
		if len(x) == 0:
			return None
		return math.sqrt(self.var(x))

if __name__ == "__main__":
	stat = TinyStatician()
	print(stat.median([26.1, 25.6, 25.7, 25.2, 25, 27.8, 24.1]))
	print(stat.quartiles([10, 25, 30, 40, 41, 42, 50, 55, 70, 101, 110, 111, 45]))
	print(stat.median([1, 42, 300, 10, 59]))
	print(stat.var([1, 42, 300, 10, 59]))
	print(stat.std([1, 42, 300, 10, 59]))
