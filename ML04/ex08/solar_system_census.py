import os, sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath('../ex00/'))
sys.path.append(os.path.abspath('../ex07'))

if __name__ == "__main__":
	df_x = pd.read_csv('resources/solar_system_census.csv')
	df_y = pd.read_csv('resources/solar_system_census_planets.csv')
	
