import os, sys
sys.path.append(os.path.abspath("../ex04/"))
from mylinearregression import MyLinearRegression as MyLR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
	df = pd.read_csv('spacecraft_data.csv')
	Aage = df['Age'].to_numpy().reshape(-1,1)
	Bthrust_pw = df['Thrust_power'].to_numpy().reshape(-1, 1)
	Cterameters = df['Terameters'].to_numpy().reshape(-1, 1)
	Dsell_price = df['Sell_price'].to_numpy().reshape(-1, 1)

	############
	# PART ONE #
	############

	myLR_age = MyLR([[1.], [1.]], alpha=0.01, max_iter=5000)
	new_theta = myLR_age.fit_(Aage, Dsell_price)
	Price_predict = myLR_age.predict_(Aage)

	plt.figure()
	plt.scatter(Aage, Dsell_price, label='Sell Price', c='#393766')
	plt.scatter(Aage, Price_predict, label='Predicted Sell Price', c='#026df7')
	plt.xlabel(r'$x_1$: age (in years)')
	plt.ylabel('y: sell price (in keuros)')
	plt.grid()
	plt.legend()
	plt.show()

	myLR_thrust = MyLR([[1.], [1.]], alpha=0.0001, max_iter=500)
	new_theta = myLR_thrust.fit_(Bthrust_pw, Dsell_price)
	Price_predict = myLR_thrust.predict_(Bthrust_pw)

	plt.figure()
	plt.scatter(Bthrust_pw, Dsell_price, label='Sell Price', c='#3e7820')
	plt.scatter(Bthrust_pw, Price_predict,
	            label='Predicted Sell Price', c='#58f705')
	plt.xlabel(r'$x_2$: thrust power (in 10Km/s)')
	plt.ylabel('y: sell price (in keuros)')
	plt.grid()
	plt.legend()
	plt.show()

	myLR_distance = MyLR([[1.], [1.]], alpha=0.0002, max_iter=100000)
	new_theta = myLR_distance.fit_(Cterameters, Dsell_price)
	Price_predict = myLR_distance.predict_(Cterameters)

	plt.figure()
	plt.scatter(Cterameters, Dsell_price, label='Sell Price', c='#852678')
	plt.scatter(Cterameters, Price_predict,
	            label='Predicted Sell Price', c='#fa00d9')
	plt.xlabel(r'$x_3$: distance totalizer value of spacecraft (in Tmeters)')
	plt.ylabel('y: sell price (in keuros)')
	plt.grid()
	plt.legend()
	plt.show()

	############
	# PART TWO #
	############

	# X = np.array(df[['Age', 'Thrust_power', 'Terameters']])
	# Y = np.array(df[['Sell_price']])
	# my_lreg = MyLR(theta=[[1.0], [1.0], [1.0], [1.0]], alpha=1e-5, max_iter=3000000)
	# print(my_lreg.mse_(my_lreg.predict_(X), Y))
	# my_lreg.fit_(X, Y)
	# print(my_lreg.theta)
	# Xpredict = my_lreg.predict_(X)
	# print(my_lreg.mse_(Xpredict,Y))

	# plt.figure()
	# plt.scatter(Aage, Dsell_price, label='Sell Price', c='#393766')
	# plt.scatter(Aage, Xpredict, label='Predicted Sell Price', c='#026df7')
	# plt.xlabel(r'$x_1$: age (in years)')
	# plt.ylabel('y: sell price (in keuros)')
	# plt.grid()
	# plt.legend()
	# plt.show()

	# plt.figure()
	# plt.scatter(Bthrust_pw, Dsell_price, label='Sell Price', c='#3e7820')
	# plt.scatter(Bthrust_pw, Xpredict,
	#             label='Predicted Sell Price', c='#58f705')
	# plt.xlabel(r'$x_2$: thrust power (in 10Km/s)')
	# plt.ylabel('y: sell price (in keuros)')
	# plt.grid()
	# plt.legend()
	# plt.show()

	# plt.figure()
	# plt.scatter(Cterameters, Dsell_price, label='Sell Price', c='#852678')
	# plt.scatter(Cterameters, Xpredict,
	#             label='Predicted Sell Price', c='#fa00d9')
	# plt.xlabel(r'$x_3$: distance totalizer value of spacecraft (in Tmeters)')
	# plt.ylabel('y: sell price (in keuros)')
	# plt.grid()
	# plt.legend()
	# plt.show()
