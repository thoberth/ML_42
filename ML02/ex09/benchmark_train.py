import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append(os.path.abspath("../ex04/"))
sys.path.append(os.path.abspath("../ex06/"))
sys.path.append(os.path.abspath("../ex08/"))
from data_spliter import data_spliter
from mylinearregression import MyLinearRegression as MyLR
from polynomial_model import *

if __name__=="__main__":
	df = pd.read_csv('space_avocado.csv')
	Yprice = df['target'].to_numpy().reshape(-1, 1)
	Xfeatures = df['weight'].to_numpy().reshape(-1, 1)
	Xfeatures = np.concatenate((Xfeatures, df['prod_distance'].to_numpy().reshape(-1, 1)), axis = 1)
	Xfeatures = np.concatenate((Xfeatures, df['time_delivery'].to_numpy().reshape(-1, 1)), axis = 1)
	for col in range(2, Xfeatures.shape[1],):
		# plt.subplot(3,2, col+1)
		plt.figure(figsize=(12, 12))
		plt.scatter(Xfeatures[:, col], Yprice, label='Ptrue', c='#0b68e0')
		for a in range(1,6):
			to_train = data_spliter(Xfeatures, Yprice, float(a / 5))
			for i in range(1, 5):
				lr = MyLR([[1.] for _ in range(i + 1)], alpha=1e-7, max_iter=100000)
				X = add_polynomial_features(to_train[0][:, 0].reshape(-1, 1), i)
				for col_polynom in range(1, to_train[0].shape[1]):
					X = np.concatenate((X, add_polynomial_features(
						to_train[0][:, col_polynom].reshape(-1, 1), i)), axis=1)
				print(X.shape)
				lr.fit_(X, to_train[2])
				Ypredict = lr.predict_(X)
				plt.plot(X[:, col], Ypredict, label=f'$Spredict_{i},{a/5}$', )
		plt.ylim([Yprice.min() - 10, Yprice.max() + 10])
		plt.xlim([Xfeatures[:, col].min() - 10, Xfeatures[:, col].max() + 10])
		plt.grid()
		plt.legend()
	plt.show()
