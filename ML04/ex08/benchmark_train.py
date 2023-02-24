import os
import sys
import numpy as np
import pandas as pd
import pickle
sys.path.append(os.path.abspath('../ex00/'))
from polynomial_features_extended import add_polynomial_features
sys.path.append(os.path.abspath('../ex07'))
from my_logistic_regression import MyLogisticRegression as MyLR
sys.path.append(os.path.abspath("../../ML02/ex08/"))
from data_spliter import data_spliter
sys.path.append(os.path.abspath('../../ML03/ex06/'))
from other_metrics import f1_score_
sys.path.append(os.path.abspath("../../ML01/ex04/"))
from z_score import zscore

def sort_y(n, y):
	for i in range(len(y)):
		if y[i] == n:
			y[i] = 1
		else:
			y[i] = 0
	return y

def save_models(results):
	file = open('models.pickle', 'wb')
	pickle.dump(results, file)
	file.close()

def best_predict(modeles, x_evaluate, y_evaluate):
	Y_HAT = []
	for i in modeles:
		Y_HAT.append(i.predict_(x_evaluate))
	y_hat = np.zeros(y_evaluate.shape)
	for i, theta_zero, theta_one, theta_two, theta_three in zip(range(len(y_hat)), Y_HAT[0], Y_HAT[1], Y_HAT[2], Y_HAT[3]):
		best = max(theta_zero, theta_one, theta_two, theta_three)
		if best == theta_zero:
			y_hat[i] = 0
		elif best == theta_one:
			y_hat[i] = 1
		elif best == theta_two:
			y_hat[i] = 2
		elif best == theta_three:
			y_hat[i] = 3
	return y_hat

def train_models(X, y, lambda_):
	perm = np.random.permutation(len(X))
	s_x = X[perm]
	s_y = y[perm]
	x_cross = np.array_split(s_x, 10)
	y_cross = np.array_split(s_y, 10)
	modeles = []
	score = []
	for i in range(len(x_cross)):
		x_fold = np.concatenate([x_cross[j] for j in range(len(x_cross)) if j != i])
		y_fold = np.concatenate([y_cross[j] for j in range(len(y_cross)) if j != i])
		x_evaluate = x_cross[i]
		y_evaluate = y_cross[i]
		modeles.clear()
		for j in range(4):
			lr = MyLR(theta=np.random.rand(
				X.shape[1] + 1, 1).reshape(-1, 1), alpha=1e-1, max_iter=10000, lambda_=lambda_)
			lr.fit_(x_fold, sort_y(j, np.copy(y_fold)))
			modeles.append(lr)
		y_hat = best_predict(modeles, x_evaluate, y_evaluate)
		print(f'Precision : {len(y_hat[y_hat == y_evaluate])} / {len(y_hat)}')
		f1score = 0.
		for k in [0, 1, 2, 3]:
			f1score += f1_score_(y_evaluate, y_hat, k)
		f1score /= len([0, 1, 2, 3])
		score.append(f1score)
	print('score F1 mean =', np.mean(score))
	return modeles, np.mean(score)

if __name__ == "__main__":
	df_x = pd.read_csv('resources/solar_system_census.csv')
	df_y = pd.read_csv('resources/solar_system_census_planets.csv')
	X = np.array(df_x[['weight', 'height', 'bone_density']])
	Y = np.array(df_y[['Origin']])

	X = add_polynomial_features(X, 3)
	for i in range(X.shape[1]):
		X[:, i] = zscore(X[:, i])
	lambda_ = [0, .2, .4, .6, .8, 1]
	models = {}
	scores = {}
	for w in range(1, 4):
		weight_poly = [0, 3, 6][:w]
		for h in range(1, 4):
			height_poly = [1, 4, 7][:h]
			for b in range(1, 4):
				bone_poly = [2, 4, 8][:b]
				for i in range(len(lambda_)):
					rank = 'w{}h{}b{}l{}'.format(w, h, b, lambda_[i])
					print(rank)
					to_train = X[:, np.concatenate((weight_poly, height_poly, bone_poly))]
					models[rank], scores[rank] = train_models(to_train, Y, lambda_[i])
	save_models({"score": scores, "models": models})
	[print(m, s) for m in models.values() for s in scores.values()]