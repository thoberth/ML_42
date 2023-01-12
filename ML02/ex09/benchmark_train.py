import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append(os.path.abspath("../ex04/"))
sys.path.append(os.path.abspath("../ex06/"))
sys.path.append(os.path.abspath("../ex08/"))
sys.path.append(os.path.abspath("../../ML01/ex04/"))
from z_score import zscore
from data_spliter import data_spliter
from mylinearregression import MyLinearRegression as MyLR
from polynomial_model import *

def	polynome_features_and_stand(x):
	X = zscore(x)
	X = add_polynomial_features(X, 4)
	return (X)

if __name__=="__main__":
	df = pd.read_csv('space_avocado.csv')
	Yprice = df['target'].to_numpy().reshape(-1, 1)
	Wfeat = df['weight'].to_numpy().reshape(-1, 1)
	Pfeat = df['prod_distance'].to_numpy().reshape(-1, 1)
	Tfeat = df['time_delivery'].to_numpy().reshape(-1, 1)
	# la standardization (zscore) et l'ajout de valeur carre (polynome) des donnees
	Wfeat = polynome_features_and_stand(Wfeat)
	Pfeat = polynome_features_and_stand(Pfeat)
	Tfeat = polynome_features_and_stand(Tfeat)
	Yprice = zscore(Yprice)
	Xfeatures = np.concatenate(
		(Wfeat, np.concatenate((Pfeat, Tfeat), axis=1)), axis=1)
	Xtrain, Xtest, Ytrain, Ytest = data_spliter(Xfeatures, Yprice, 0.5)
	Bmse = None
	i = 0
	fig, ax = plt.subplots()
	plt.grid()
	for a in range(1, Wfeat.shape[1] + 1):
		for b in range(1, Pfeat.shape[1] + 1):
			for c in range(1, Tfeat.shape[1] + 1):
				lr = MyLR([[1.] for _ in range( a + b + c + 1)], alpha=1e-1, max_iter=2000)
				X = np.concatenate((Xtrain[:, :a],
					np.concatenate((Xtrain[:, 4:4+b], Xtrain[:, 8:8+c]), axis = 1)), axis=1)
				lr.fit_(X, Ytrain)
				Xto_test = np.concatenate((Xtest[:, :a],
					np.concatenate((Xtest[:, 4:4+b], Xtest[:, 8:8+c]), axis=1)), axis=1)
				Ypredict = lr.predict_(Xto_test)
				mse = lr.mse_(Ytest, Ypredict)
				if not (math.isinf(mse) and math.isnan(mse)):
					i+=1
					ax.scatter(i, mse)
				if Bmse == None or Bmse[3] > mse and not math.isnan(mse):
					Bmse = [[a], [b], [c], [mse], [lr]]
				print(f'for a p={a}, b p={b}, c p={c} mse={mse}')
	print(f'\n\nwe found that the best models is a p={Bmse[0]},\
 b p={Bmse[1]}, c p={Bmse[2]} with mse={float(Bmse[3][0])}')
	plt.show()