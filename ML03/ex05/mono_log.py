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
	if check_arg(sys.argv) == False:
		exit()
	try:
		zipcode = int(check_arg(sys.argv))
		if zipcode > 3 or zipcode < 0:
			raise Exception()
	except:
		print('Error with value of "-zipcode" value must be an int between 0 - 3')

#### 1 SPLIT DATASET INTO TRAINING AND TEST SET
	data = pd.read_csv("solar_system_census.csv")
	data_planet = pd.read_csv("solar_system_census_planets.csv")
	# data_planet.drop(data_planet.columns[data_planet.columns.str.contains('Unnamed', case = False)], axis=1, inplace=True)
	
