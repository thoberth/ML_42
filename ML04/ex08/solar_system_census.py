from benchmark_train import *

def train_models(X, y, lambda_):
	lr = MyLR(theta=np.random.rand(
				X.shape[1] + 1, 1).reshape(-1, 1), max_iter=15000, lambda_=lambda_)
	print(lr.loss_(lr.predict_(X), y))
	lr.fit_(X, y)
	print(lr.loss_(lr.predict_(X), y))

if __name__ == "__main__":
	df_x = pd.read_csv('resources/solar_system_census.csv')
	df_y = pd.read_csv('resources/solar_system_census_planets.csv')
	X = np.array(df_x[['weight', 'height', 'bone_density']])
	y = np.array(df_y[['Origin']])

	X = add_polynomial_features(X, 3)
	for i in range(X.shape[1]):
		X[:, i] = zscore(X[:, i])

	x_train, x_test, y_train, y_test = data_spliter(X, y, 0.7)

	file = open("models.pickle", "rb") # "rb" -> read in binary mode
	models = pickle.load(file)
	file.close()

	best = None
	index = 0
	for i, (key, value) in enumerate(models['score'].items()):
		if best == None:
			best = {key: value}
		elif list(best.values()) < value:
			best = {key: value}
			index = i
	print('best is ', best, index)
	res = str(list(best.keys())[0])
	w, h, b, lambda_ = int(res[1]), int(res[3]), int(res[5]), float(res[7:])
	to_train = x_train[:, np.concatenate(([0, 3, 6][:w], [1, 4, 7][:h], [2, 5, 8][:b]))]
	train_models(to_train, y_train, lambda_)