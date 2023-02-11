import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.abspath("../ex04/"))
from my_logistic_regression import MyLogisticRegression as MyLR
sys.path.append(os.path.abspath("../../ML02/ex08/"))
from data_spliter import data_spliter
import numpy as np
import pandas as pd

def check_arg(argv):
	for i in argv:
		if i.find('-zipcode') != -1:
			return i[len('-zipcode='):]
	print('Error, "zipcode" argument is missing.')
	return False

if __name__ == "__main__":
#### 1 TAKE ZIPCODE AS ARGUMENT BETWEEN 0 - 3
	if check_arg(sys.argv) == False:
		exit()
	try:
		zipcode = int(check_arg(sys.argv))
		if zipcode > 3 or zipcode < 0:
			raise Exception()
	except:
		print('Error with value of "-zipcode" value must be an int between 0 - 3')

#### 2 SPLIT DATASET INTO TRAINING AND TEST SET
#### 3 LABEL EACH CITIZEN: 1=FROM OUR PLANET, 0=ALL OTHERS
	data = pd.read_csv("resources/solar_system_census.csv")
	data_planet = pd.read_csv("resources/solar_system_census_planets.csv")
	# data_planet.drop(data_planet.columns[data_planet.columns.str.contains('Unnamed', case = False)], axis=1, inplace=True)
	X = np.array(data[['weight', 'height', 'bone_density']])
	Y = np.array(data_planet[['Origin']])
	for i in range(len(Y)):
		if Y[i] == zipcode:
			Y[i] = 1
		else:
			Y[i] = 0
	x_train, x_test, y_train, y_test = data_spliter(X, Y, 0.5)
	# print(x_train, x_test, y_train, y_test)
	mylr = MyLR(np.random.rand(x_train.shape[1] + 1, 1).reshape(-1, 1), alpha=1e-4, max_iter=25000)
	Y_hat = mylr.predict_(x_train)
	print('La perte est de {},\nOn entraine notre modele....'.format(mylr.loss_(y_train, Y_hat)))
#### 4 TRAIN OUR MODEL TO PREDICT ON TRAIN SET
	mylr.fit_(x_train, y_train)
	Y_hat = mylr.predict_(x_train)
	print('La perte est maintenant de {}'.format(
		mylr.loss_(y_train, Y_hat)))
#### 5 CALCULATE AND DISPLAY THE FRACTION OF TOTAL PREDICTION
	y_hat_test = mylr.predict_(x_test)
	y_hat_test[y_hat_test >= 0.5] = 1
	y_hat_test[y_hat_test < 0.5] = 0
	count = 0
	for i in range(len(y_hat_test)):
		if y_hat_test[i] == y_test[i]:
			count += 1
	print('On predit avec les test set et on obtient:', (count*100)/len(y_hat_test), '% de valeur exacte')
#### 6 plot 3 scatter
	plt.figure()
	plt.scatter(x_test[:, 0], y_hat_test,
	            label='prediction', c='#51348a', alpha=0.5)
	plt.scatter(x_test[:, 0], y_test, label='vraie valeur',
	            c='#32a852', alpha=0.5)
	plt.xlabel('Weight')
	plt.ylabel('1 = is from our Planet,\n0 = is not from our Planet')
	plt.legend()
	plt.show()

	plt.figure()
	plt.scatter(x_test[:, 1], y_hat_test,
	            label='prediction', c='#51348a', alpha=0.5)
	plt.scatter(x_test[:, 1], y_test, label='vraie valeur',
	            c='#32a852', alpha=0.5)
	plt.legend()
	plt.xlabel('Height')
	plt.ylabel('1 = is from our Planet,\n0 = is not from our Planet')
	plt.show()

	plt.figure()
	plt.scatter(x_test[:, 2], y_hat_test,
	            label='prediction', c='#51348a', alpha=0.5)
	plt.scatter(x_test[:, 2], y_test, label='vraie valeur',
	            c='#32a852', alpha=0.5)
	plt.xlabel('Bone Density')
	plt.ylabel('1 = is from our Planet,\n0 = is not from our Planet')
	plt.legend()
	plt.show()
