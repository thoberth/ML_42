import math
import numpy as np

class TinyStatician:
	def __init__(self):
		pass

	def	mean(self, x):
		if len(x) == 0 or not (isinstance(x, list) or isinstance(x, np.ndarray)):
			return None
		r = 0
		for i in x:
			r+=i
		r/=len(x)
		return float(r)

	def median(self, x):
		if len(x) == 0 or not (isinstance(x, list) or isinstance(x, np.ndarray)):
			return None
		if isinstance(x, np.ndarray):
			x = x.tolist()
		x.sort()
		r = len(x)
		if r % 2 == 0:
			r = x[(int(r/2)) - 1]
		else :
			r = x[(int((r+1)/2)) - 1]
		return float(r)

	def quartile(self, x):
		if len(x) == 0 or not (isinstance(x, list) or isinstance(x, np.ndarray)):
			return None
		if isinstance(x, np.ndarray):
			x = x.tolist()
		x.sort()
		q1 = x[math.ceil((len(x)/4)) - 1]
		q3 = x[math.ceil((3*len(x)/4)) - 1]
		return [float(q1), float(q3)]

	def percentile(self, x, p):
		if len(x) == 0 or not ((isinstance(x, list) or isinstance(x, np.ndarray)) and (isinstance(p, int) or isinstance(p, float))):
			return None
		if isinstance(x, np.ndarray):
			x = x.tolist()
		x.sort()
		rang  = (p/100*(len(x)-1))+1
		rang_ordinal = math.floor(rang)
		decimal_part = rang - rang_ordinal
		centile = x[rang_ordinal-1] + decimal_part*(x[rang_ordinal] - x[rang_ordinal-1])
		return centile

	def var(self, x):
		if len(x) == 0 or not (isinstance(x, list) or isinstance(x, np.ndarray)):
			return None
		if isinstance(x, np.ndarray):
			x = x.tolist()
		x.sort()
		r = 0.0
		mean = self.mean(x) 
		for i in x:
			r += math.pow(i - mean, 2)
		return(r / (len(x) -1))

	def std(self, x):
		if len(x) == 0 or not (isinstance(x, list) or isinstance(x, np.ndarray)):
			return None
		return math.sqrt(self.var(x))

if __name__ == "__main__":
	stat = TinyStatician()
	a = [1, 42, 300, 10, 59]
	print(stat.median([26.1, 25.6, 25.7, 25.2, 25, 27.8, 24.1]))
	print(stat.quartile([10, 25, 30, 40, 41, 42, 50, 55, 70, 101, 110, 111, 45]))
	print(stat.median(a))
	print(stat.var(a))
	print(stat.std(a))
	print(stat.percentile(a, 10))
	print(stat.percentile(a, 15))
	print(stat.percentile(a, 20))
	stat.percentile([2, 4, 8, 9, 11, 13, 15, 17, 20, 24,  29, 30],40)
