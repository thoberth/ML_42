import os, sys
sys.path.append(os.path.abspath('../..'))
from ML04.ex00.polynomial_features_extended import add_polynomial_features
import numpy as np
from ML04.ex06.ridge import MyRidge
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from ML01.ex04.z_score import zscore

def save_models(results):
	file = open('models.pickle', 'wb')
	pickle.dump(results, file)
	file.close()

def train_model(X, y, lambda_):
	perm = np.random.permutation(len(X))
	s_x = X[perm]
	s_y = y[perm]
	x_cross = np.array_split(s_x, 4)
	y_cross = np.array_split(s_y, 4)
	loss = []
	for i in range(len(x_cross)):
		x_fold = np.concatenate([x_cross[j] for j in range(len(x_cross)) if j != i])
		y_fold = np.concatenate([y_cross[j] for j in range(len(y_cross)) if j != i])
		x_evaluate = x_cross[i]
		y_evaluate = y_cross[i]
		lr = MyRidge(thetas=np.random.rand(
			X.shape[1] + 1, 1).reshape(-1, 1), alpha=1e-2, max_iter=10000, lambda_=lambda_)
		lr.fit_(x_fold, y_fold)
		y_hat = lr.predict_(x_evaluate)
		loss.append(lr.loss_(y_evaluate, y_hat))
	print(np.mean(loss))

def main():
	df = pd.read_csv('resources/space_avocado.csv')
	X = np.array(df[['weight', 'prod_distance', 'time_delivery']])
	y = np.array(df[['target']])

	X = add_polynomial_features(X, 4)
	for i in range(X.shape[1]):
		X[:, i] = zscore(X[:, i])
	lambda_ = np.linspace(0., 1., 5, endpoint=False)

	for w in range(1, 5):
		w_poly = [0, 3, 6, 9][:w]
		for p in range(1, 5):
			p_poly = [1, 4, 7, 10][:p]
			for t in range(1, 5):
				t_poly = [2, 5, 8, 11][:t]
				for i in lambda_:
					model_ref = f'w{w}p{p}t{t}l{i:.1f}'
					to_train = X[:, np.concatenate((w_poly, p_poly, t_poly))]
					train_model(to_train, y, i)
		exit()

if __name__ == "__main__":
	main()
