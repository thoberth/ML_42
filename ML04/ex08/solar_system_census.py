from benchmark_train import *
import matplotlib.pyplot as plt

def render(x, y_hat, y, f1score, prec):
	fig = plt.figure(figsize=(13, 10))
	ax = fig.add_subplot(projection='3d')
	for xi, yi, y_hati in zip(x, y, y_hat):
		if yi == y_hati:
			m = 'o'
			c = 'green'
		else:
			m = 'x'
			c = 'red'
		ax.scatter(xi[0], xi[1], xi[2], marker=m, c=c)
	ax.set_xlabel('Weight')
	ax.set_ylabel('Height')
	ax.set_zlabel('Bone Density')
	plt.show()

def train_models(x_train, x_test, y_train, y_test, lambda_):
	models = []
	for i in range(4):
		lr = MyLR(theta=np.random.rand(
					x_train.shape[1] + 1, 1).reshape(-1, 1), alpha=1e-1, max_iter=10000, lambda_=lambda_)
		lr.fit_(x_train, sort_y(i, np.copy(y_train)))
		models.append(lr)
	y_hat = best_predict(models, x_test, y_test)
	prec = f'{len(y_hat[y_hat == y_test])} / {len(y_hat)}'
	print(f'Precision : {prec}')
	f1score = 0.
	for i in range(4):
		f1score += f1_score_(y_test, y_hat, i)
	f1score /= 4
	print(f'F1 score is {f1score}')
	return y_hat, f1score, prec

if __name__ == "__main__":
	df_x = pd.read_csv('resources/solar_system_census.csv')
	df_y = pd.read_csv('resources/solar_system_census_planets.csv')
	df_x = np.array(df_x[['weight', 'height', 'bone_density']])
	df_y = np.array(df_y[['Origin']])

	df_x = add_polynomial_features(df_x, 3)
	x_train, x_test, y_train, y_test = data_spliter(df_x, df_y, 0.7)
	x_train_norm = np.copy(x_train)
	for i in range(x_train.shape[1]):
		x_train_norm[:, i] = zscore(x_train[:, i])
	x_test_norm = np.copy(x_test)
	for i in range(x_test.shape[1]):
		x_test_norm[:, i] = zscore(x_test[:, i])

	file = open("models.pickle", "rb") # "rb" -> read in binary mode
	models = pickle.load(file)
	file.close()

	best = None
	fig, ax = plt.subplots(figsize=(17, 10))
	for i, (key, value) in enumerate(models['score'].items()):
		ax.scatter(key, value)
		if best == None:
			best = {key: value}
		elif list(best.values()) < value:
			best = {key: value}
	ax.set_xticks([])
	title = 'best model is ' + str(list(best.keys())[0]) + ' with a f1_score of ' + f'{list(best.values())[0]:.4f}'
	plt.title(title)
	plt.ylabel('F1 scores:')
	plt.xlabel('Models')
	plt.grid()
	plt.show()
	res = str(list(best.keys())[0])
	w, h, b, lambda_ = int(res[1]), int(res[3]), int(res[5]), float(res[7:])
	x_train_norm = x_train_norm[:, np.concatenate(
		([0, 3, 6][:w], [1, 4, 7][:h], [2, 5, 8][:b]))]
	x_test_norm = x_test_norm[:, np.concatenate(
		([0, 3, 6][:w], [1, 4, 7][:h], [2, 5, 8][:b]))]
	y_hat, f1score, prec = train_models(x_train_norm, x_test_norm, y_train, y_test, lambda_)
	render(x_test, y_hat, y_test, f1score, prec)
