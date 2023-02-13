import matplotlib.pyplot as plt
import sys, os
import numpy as np
sys.path.append(os.path.abspath("../ex04/"))
from my_logistic_regression import MyLogisticRegression as MyLR
sys.path.append(os.path.abspath("../../ML02/ex08/"))
sys.path.append(os.path.abspath("../../ML01/ex04/"))
from z_score import zscore
from data_spliter import data_spliter
import pandas as pd

def sort_y(n, y):
	for i in range(len(y)):
		if y[i] == n:
			y[i] = 1
		else:
			y[i] = 0
	return y

if __name__=="__main__":
	df_x = pd.read_csv('resources/solar_system_census.csv')
	df_y = pd.read_csv('resources/solar_system_census_planets.csv')
	df_x = np.array(df_x[['weight', 'height', 'bone_density']])
	df_y = np.array(df_y[['Origin']])
	for i in range(df_x.shape[1]):
		df_x[:, i] = zscore(df_x[:, i])
	x_train, x_test, y_train, y_test = data_spliter(df_x, df_y, 0.8)
	thetas_lst = []
	for i in range(4):
		Y = sort_y(i, np.copy(y_train))
		reg = MyLR(np.random.rand(x_train.shape[1] + 1, 1).reshape(-1, 1), max_iter=75000)
		reg.fit_(x_train, Y)
		thetas_lst.append(reg)
	pred = []
	for i in thetas_lst:
		pred.append(i.predict_(x_test))
	y_hat = np.zeros(y_test.shape)
	for i, theta_zero, theta_one, theta_two, theta_three in zip(range(len(y_hat)), pred[0], pred[1], pred[2], pred[3]):
		best = max(theta_zero, theta_one, theta_two, theta_three)
		if best == theta_zero:
			y_hat[i] = 0
		elif best == theta_one:
			y_hat[i] = 1
		elif best == theta_two:
			y_hat[i] = 2
		elif best == theta_three:
			y_hat[i] = 3

	print(f'Precision : {len(y_hat[y_hat == y_test])} / {len(y_hat)}')

	fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
	fig.set_size_inches(12.5, 5.5)
	fig.tight_layout()
	ax1.scatter(x_test[:,0], y_test, label="Real values")
	ax1.scatter(x_test[:, 0], y_hat, label="Predictions", alpha=0.5)
	ax1.grid()
	ax1.legend()
	ax1.set_xlabel("weight")
	ax1.set_ylabel("Origin")
	ax2.scatter(x_test[:,1], y_test, label="Real values")
	ax2.scatter(x_test[:,1], y_hat, label="Predictions", alpha=0.5)
	ax2.grid()
	ax2.legend()
	ax2.set_xlabel("height")
	ax2.set_ylabel("Origin")
	ax3.scatter(x_test[:,2], y_test, label="Real values")
	ax3.scatter(x_test[:, 2], y_hat, label="Predictions", alpha=0.5)
	ax3.grid()
	ax3.legend()
	ax3.set_xlabel("bone_density")
	ax3.set_ylabel("Origin")
	fig.suptitle("Predictions comparisions")
	plt.show()