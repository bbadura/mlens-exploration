import sys
import pickle
from operator import itemgetter

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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

# Model evaluation
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator

# ML Ensemble
from mlens.ensemble import SuperLearner
from mlens.ensemble import Subsemble
from mlens.ensemble import BlendEnsemble
from mlens.ensemble import SequentialEnsemble

# Output
from texttable import Texttable

seed = 9880
iters = 5

#Adds an ensemble composed of the same elements with different (randomized) parameters
def add_superlearner(name, models, X_train, Y_train, X_test, Y_test):
	# Establish and reset variables
	acc_score_cv = None
	acc_score = None
	time_ = None
	ensemble = SuperLearner(scorer=accuracy_score, random_state=seed)

	ensemble.add(models)
	# Attach the final meta estimator
	ensemble.add_meta(SVC())

	start = time.time()
	ensemble.fit(X_train, Y_train)
	preds = ensemble.predict(X_test)
	acc_score = accuracy_score(preds, Y_test)
	end = time.time()
	time_ = end - start

	return {"Ensemble": name, "Meta_Classifier": "SVC", "Accuracy_Score": acc_score, "Runtime": time_}

def add_subsemble(name, models, X_train, Y_train, X_test, Y_test):
	# Establish and reset variables
	acc_score_cv = None
	acc_score = None
	time_ = None
	ensemble = Subsemble(scorer=accuracy_score, random_state=seed)

	ensemble.add(models)
	# Attach the final meta estimator
	ensemble.add(SVC(), meta=True)

	start = time.time()
	ensemble.fit(X_train, Y_train)
	preds = ensemble.predict(X_test)
	acc_score = accuracy_score(preds, Y_test)
	end = time.time()
	time_ = end - start

	return {"Ensemble": name, "Meta_Classifier": "SVC", "Accuracy_Score": acc_score, "Runtime": time_}

def add_blend(name, models, X_train, Y_train, X_test, Y_test):
	# Establish and reset variables
	acc_score_cv = None
	acc_score = None
	time_ = None
	ensemble = BlendEnsemble(scorer=accuracy_score, random_state=seed)

	ensemble.add(models)
	# Attach the final meta estimator
	ensemble.add_meta(SVC())

	start = time.time()
	ensemble.fit(X_train, Y_train)
	preds = ensemble.predict(X_test)
	acc_score = accuracy_score(preds, Y_test)
	end = time.time()
	time_ = end - start

	return {"Ensemble": name, "Meta_Classifier": "SVC", "Accuracy_Score": acc_score, "Runtime": time_}

def add_sequential(name, models, X_train, Y_train, X_test, Y_test):
	# Establish and reset variables
	acc_score_cv = None
	acc_score = None
	time_ = None
	ensemble = SequentialEnsemble(scorer=accuracy_score, random_state=seed)

	# Add a subsemble with 5 partitions as first layer
	ensemble.add('subsemble', models, partitions=10, folds=10)

	# Add a super learner as second layer
	ensemble.add('stack', models, folds=20)
	ensemble.add_meta(SVC())

	start = time.time()
	ensemble.fit(X_train, Y_train)
	preds = ensemble.predict(X_test)
	acc_score = accuracy_score(preds, Y_test)
	end = time.time()
	time_ = end - start

	return {"Ensemble": name, "Meta_Classifier": "SVC", "Accuracy_Score": acc_score, "Runtime": time_}


# Runs the program... add test datasets to this portion
def main():
	#read in data and parse
	files = ['data/iris.csv']
	train_df = pd.read_csv(files[0], header=None)
	file_output = "output/output_iris_new.txt"

	#map classifier as binary
	train_df[4] = train_df[4].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}).astype(int)

	#separate models
	X_train = train_df.drop(4, axis=1)
	Y_train = train_df[4]

	X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size=0.75, test_size=0.25)

	#feature selection (currently only works on datasets that do not have named index fields)
	# selector = SelectKBest(f_classif, k=20)
	# X_train = selector.fit_transform(X_train, Y_train)
	# X_test = X_test[selector.get_support(indices=True)]

	output = [0] * iters
	#print("------Feature Selection Complete------")
	for i in range(iters):
		models = []
		for j in range(0,15):
			#try out a new classifier
			pipeline1 = Pipeline([
				('dtc', DecisionTreeClassifier(max_features=random.randint(1,4),max_depth=random.randint(1,200),random_state=random.randint(1,5000)))
			])
			models.append(pipeline1)

		output[i] = {}

		# Function calls to create and test ensembles
		output[i]['super_dtc'] = add_superlearner('super_dtc', models, X_train, Y_train, X_test, Y_test)
		print("---------------  10%  ---------------")
		output[i]['sub_dtc'] = add_subsemble('sub_dtc', models, X_train, Y_train, X_test, Y_test)
		print("---------------  20%  ---------------")
		output[i]['blend_dtc'] = add_blend('blend_dtc', models, X_train, Y_train, X_test, Y_test)
		print("---------------  30%  ---------------")

		models = []
		for j in range(0,15):
			#try out a new classifier
			pipeline1 = Pipeline([
				('knc', KNeighborsClassifier(n_neighbors=random.randint(1,20)))
			])
			models.append(pipeline1)

		output[i]['super_knc'] = add_superlearner('super_knc', models, X_train, Y_train, X_test, Y_test)
		print("---------------  40%  ---------------")
		output[i]['sub_knc'] = add_subsemble('sub_knc', models, X_train, Y_train, X_test, Y_test)
		print("---------------  50%  ---------------")
		output[i]['blend_knc'] = add_blend('blend_knc', models, X_train, Y_train, X_test, Y_test)
		print("---------------  60%  ---------------")

		# models = []
		# for j in range(0,10):
		# 	#try out a new classifier
		# 	pipeline1 = Pipeline([
		# 		('ada', AdaBoostClassifier(n_estimators=random.randint(50,150),random_state=random.randint(1,5000)))
		# 	])
		# 	models.append(pipeline1)

		# output[i]['super_ada'] = add_superlearner('super_ada', models, X_train, Y_train, X_test, Y_test)
		# print("---------------  70%  ---------------")
		# output[i]['sub_ada'] = add_subsemble('sub_ada', models, X_train, Y_train, X_test, Y_test)
		# print("---------------  80%  ---------------")
		# output[i]['blend_ada'] = add_blend('blend_ada', models, X_train, Y_train, X_test, Y_test)
		# print("---------------  90%  ---------------")

	t = Texttable()
	average_acc = {}
	average_time = {}
	t.add_row(['Dataset', 'Ensemble', 'Meta Classifier', 'Accuracy Score', 'Runtime'])
	for i in range(iters):
		for key, value in output[i].iteritems():
			t.add_row([key, output[i][key]["Ensemble"], output[i][key]["Meta_Classifier"], output[i][key]["Accuracy_Score"], output[i][key]["Runtime"]])
			if (i == 0):
				average_acc[key] = output[i][key]["Accuracy_Score"]
				average_time[key] = output[i][key]["Runtime"]
			else:
				average_acc[key] = average_acc[key] + output[i][key]["Accuracy_Score"]
				average_time[key] = average_time[key] + output[i][key]["Runtime"]
	for key, value in average_acc.iteritems():
		t.add_row(["Average", key, "SVC", value/iters, average_time[key]/iters])

	print(t.draw())

	if (file_output != None):
		print("Saving output to: {}".format(file_output))
		orig_stdout = sys.stdout
		f = open(file_output, 'w')
		sys.stdout = f

		print(t.draw())

		sys.stdout = orig_stdout
		f.close()


if __name__== "__main__":
	main()
