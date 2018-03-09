# data analysis and wrangling
import pandas as pd
import numpy as np
import random
import math
import time

# machine learning
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomTreesEmbedding

# Model evaluation
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator

# Ensemble
from mlens.ensemble import SuperLearner

# Output
from texttable import Texttable

def load_data(label, train, test=None):
	#read in data and parse
	train_df = pd.read_csv(train, header=None)
	if test != None:
		test_df = pd.read_csv(test, header=None)
		combine = [train_df, test_df]
	else:
		combine = [train_df]

	return combine

def main():
	datasets = {}
	datasets['first'] = load_data(label='first', train='obtrain.csv', test='obtest.csv')

	#separate models
	X_train = []
	Y_train = []
	X_test = []
	Y_test = []

	X_train.append(datasets['first'][0].drop(559, axis=1))
	Y_train.append(datasets['first'][0].drop(559, axis=1))
	X_test.append(datasets['first'][0].drop(559, axis=1))
	Y_test.append(datasets['first'][0].drop(559, axis=1))

	selector = SelectKBest(f_classif, k=150)
	X_train[0] = selector.fit_transform(X_train[0], Y_train[0])
	X_test[0] = X_test[0][selector.get_support(indices=True)]

	output = {}



if __name__ == '__main__':
	main()