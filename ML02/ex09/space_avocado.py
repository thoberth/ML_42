import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import pickle
sys.path.append(os.path.abspath("../ex04/"))
sys.path.append(os.path.abspath("../ex06/"))
sys.path.append(os.path.abspath("../../ML01/ex04/"))
from z_score import zscore
from mylinearregression import MyLinearRegression as MyLR
from polynomial_model import *


def polynome_features_and_stand(x, i):
	X = add_polynomial_features(x, i)
	for i in range(X.shape[1]):
		X[:, i] = zscore(X[:, i])
	return (X)

if __name__=="__main__":
	df = pd.read_csv('space_avocado.csv')
	Yprice = df['target'].to_numpy().reshape(-1, 1)
	Wfeat = df['weight'].to_numpy().reshape(-1, 1)
	Pfeat = df['prod_distance'].to_numpy().reshape(-1, 1)
	Tfeat = df['time_delivery'].to_numpy().reshape(-1, 1)

	f = open('models.pickle', 'rb')
	bm = pickle.load(f) # best model
	Xfeatures = np.concatenate(
		(polynome_features_and_stand(Wfeat, bm[0]), np.concatenate((polynome_features_and_stand(Pfeat, bm[1]),
			polynome_features_and_stand(Tfeat, bm[2])), axis=1)), axis=1)
	Ypredict = bm[4].predict_(Xfeatures)
	print(bm[4].mse_(zscore(Yprice), Ypredict))

	plt.subplot(2, 2, 1)
	plt.scatter(Wfeat, zscore(Yprice), label='Sell Price', c='#274edb')
	plt.scatter(Wfeat, Ypredict, label='Predicted Sell Price', c='#9949d6')
	plt.ylabel('Price')
	plt.xlabel('Weight')

	plt.subplot(2, 2, 2)
	plt.scatter(Pfeat, zscore(Yprice), label='Sell Price', c='#274edb')
	plt.scatter(Pfeat, Ypredict, label='Predicted Sell Price', c='#9949d6')
	plt.ylabel('Price')
	plt.xlabel('Prod distance')

	plt.subplot(2, 2, 3)
	plt.scatter(Tfeat, zscore(Yprice), label='Sell Price', c='#274edb')
	plt.scatter(Tfeat, Ypredict, label='Predicted Sell Price', c='#9949d6')
	plt.ylabel('Price')
	plt.xlabel('Time delivery')

	plt.grid()
	plt.legend()
	plt.show()