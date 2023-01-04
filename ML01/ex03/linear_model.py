import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.abspath("../ex02/"))
from my_linear_regression import MyLinearRegression as MyLR
import numpy as np
import pandas as pd

if __name__ == "__main__":
	thetas = np.array([[2], [0.7]])
	lr = MyLR(thetas, max_iter=100000)
	df = pd.read_csv('are_blue_pills_magics.csv')
	Yscore = pd.DataFrame.to_numpy(df.drop(['Patient', 'Micrograms'], axis=1))
	Xpill = pd.DataFrame.to_numpy(df.drop(['Score', 'Patient'], axis=1))
	lr.fit_(Xpill, Yscore)
	Y_model1 = lr.predict_(Xpill)
	plt.figure()
	plt.scatter(Xpill, Yscore, label= 'Strue(pills)')
	plt.plot(Xpill, Y_model1, label='Spredict(pills)', linestyle='--', c='C2', marker='x')
	plt.ylabel('Space driving score')
	plt.xlabel('Quantity of blue pill (in micrograms)')
	plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
			mode="expand", borderaxespad=0, ncol=2)
	plt.grid()
	plt.show()

	plt.figure()
	plt.ylabel('cost function ' +  r'J($\theta_0, \theta_1$)')
	plt.xlabel(r'$\theta_1$')

	npoints = 100
	thetas_0 = np.linspace(80, 100, 6)
	thetas_1 = np.linspace(-15, -4, npoints)
	lr2 = MyLR(np.array([[0.], [0.]]))
	for t0 in thetas_0:
		lr2.thetas[0][0] = t0
		y_cost = [0] * npoints
		for i, t1 in enumerate(thetas_1):
			lr2.thetas[1][0] = t1
			y_hat = lr2.predict_(Xpill)
			y_cost[i] = lr2.loss_(Yscore, y_hat) * 2
		plt.plot(thetas_1, y_cost, label=f"J$(\\theta_0={t0}, \\theta_1)$")
	plt.grid()
	plt.ylim([10, 150])
	plt.legend()
	plt.show()