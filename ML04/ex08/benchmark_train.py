import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath('../ex00/'))
from polynomial_features_extended import add_polynomial_features
sys.path.append(os.path.abspath('../ex07'))
from my_logistic_regression import MyLogisticRegression as MyLR
sys.path.append(os.path.abspath("../../ML02/ex08/"))
from data_spliter import data_spliter

def train_models(X, y, lambda_):
	X = np.array_split(X, 5)
	y = np.array_split(y, 5)
	for i in range(len(X)):
		x_fold = np.concatenate([X[j] for j in range(len(X)) if j != i ])
		y_fold = np.concatenate([y[j] for j in range(len(y)) if j != i])
		print(x_fold.shape[0] == y_fold.shape[0])



if __name__ == "__main__":
	df_x = pd.read_csv('resources/solar_system_census.csv')
	df_y = pd.read_csv('resources/solar_system_census_planets.csv')
	X = np.array(df_x[['weight', 'height', 'bone_density']])
	Y = np.array(df_y[['Origin']])
	x_train, x_test, y_train, y_test = data_spliter(X, Y, 0.8)

	X = add_polynomial_features(x_train, 3)
	lambda_ = [0, .2, .4, .6, .8]
	models = []
	for i in range(len(lambda_)):
		models.append(train_models(X, y_train, lambda_[i]))
