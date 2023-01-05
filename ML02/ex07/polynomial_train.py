import os, sys
sys.path.append(os.path.abspath("../ex04/"))
sys.path.append(os.path.abspath("../ex06/"))
from polynomial_model import *
from mylinearregression import MyLinearRegression as MyLR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
	df = pd.read_csv('are_blue_pills_magics.csv')
	Xmicrograms = df['Micrograms'].to_numpy().reshape(-1, 1)
	Yscore = df['Score'].to_numpy().reshape(-1, 1)
	X = Xmicrograms
	theta4 = np.array([[-20], [160], [-80], [10], [-1]]).reshape(-1, 1)
	theta5 = np.array([[1140], [-1850], [1110], [-305], [40], [-2]]).reshape(-1, 1)
	theta6 = np.array([[9110], [-18015], [13400], [-4935],
                  [966], [-96.4], [3.86]]).reshape(-1, 1)
	plt.figure()
	plt.scatter(Xmicrograms, Yscore, label='Strue', c='#0b68e0')
	for i in range(1, 7):
		if (i < 4):
			lr = MyLR([[1.]for _ in range(i + 1)], alpha=1e-5, max_iter=300000)
		elif i == 4:
			lr = MyLR(theta4, alpha=1e-7, max_iter=100000)
		elif i == 5:
			lr = MyLR(theta5, alpha=1e-8, max_iter=100000)
		elif i == 6:
			lr = MyLR(theta6, alpha=1e-9, max_iter=100000)
		X = add_polynomial_features(Xmicrograms, i)
		lr.fit_(X, Yscore)

		continuous_x = np.arange(Xmicrograms.min(), Xmicrograms.max(), 0.01).reshape(-1, 1)
		X_ = add_polynomial_features(continuous_x, i)
		YpredictLong = lr.predict_(X_)

		Ypredict = lr.predict_(X)
		mse_ = lr.loss_(Yscore, Ypredict)
		plt.plot(continuous_x, YpredictLong, label=f'$Spredict_{i}$', )
		print('mse {} = {}'.format(i, np.sum(mse_) / i))
	plt.ylim([Yscore.min() -10, Yscore.max() + 10])
	plt.grid()
	plt.legend()
	plt.show()