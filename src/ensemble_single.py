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

# Model evaluation
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator

# Ensemble
from mlens.ensemble import SuperLearner

# Output
from texttable import Texttable

seed = 9880
#listToSave = []
try:
	with open('library.pkl', 'rb') as f:
		listToSave = pickle.load(f)
except:
	listToSave = []

#Adds an ensemble composed of the same elements with different (randomized) parameters
def add_ensemble_same(name, element, meta_name, meta, cross_val, X_train, Y_train, X_test, Y_test):
	# Establish and reset variables
	acc_score_cv = None
	acc_score = None
	time_ = None
	ensemble_cv = SuperLearner(scorer=accuracy_score, random_state=seed)
	ensemble = SuperLearner(scorer=accuracy_score, random_state=seed)
	models = []

	for j in range(0,30):
		#try out a new classifier
		pipeline1 = Pipeline([
			(name, element)
		])
		models.append(pipeline1)

	if (cross_val == True):
		ensemble_cv.add(models)
		ensemble.add(models)

		# Attach the final meta estimator
		ensemble_cv.add_meta(meta)
		ensemble.add_meta(meta)

		ensemble_cv.fit(X_train[:1600], Y_train[:1600])
		preds_cv = ensemble_cv.predict(X_test[:1600])
		acc_score_cv = accuracy_score(preds_cv, Y_test[:1600])

		start = time.time()
		ensemble.fit(X_train, Y_train)
		preds = ensemble.predict(X_test)
		acc_score = accuracy_score(preds, Y_test)
		end = time.time()
		time_ = end - start
		if(acc_score > 0.83):
			if (len(listToSave) == 5):
				for x in listToSave:
					if x[1] < acc_score:
						x = [ensemble, acc_score]
						break;
			else:
				listToSave.append([ensemble, acc_score])
		listToSave.sort(key=itemgetter(1))

		return {"Ensemble_Elements": name, "Meta_Classifier": meta_name, "Accuracy_Score_CV": acc_score_cv, "Accuracy_Score": acc_score, "Runtime": time_}
	else:
		ensemble.add(models)
		# Attach the final meta estimator
		ensemble.add_meta(meta)

		start = time.time()
		ensemble.fit(X_train, Y_train)
		preds = ensemble.predict(X_test)
		acc_score = accuracy_score(preds, Y_test)
		end = time.time()
		time_ = end - start
		if(acc_score > 0.83):
			if (len(listToSave) == 5):
				for x in listToSave:
					if x[1] < acc_score:
						x = [ensemble, acc_score]
						break;
			else:
				listToSave.append([ensemble, acc_score])
		listToSave.sort(key=itemgetter(1))

		return {"Ensemble_Elements": name, "Meta_Classifier": meta_name, "Accuracy_Score_CV": "N/A", "Accuracy_Score": acc_score, "Runtime": time_}

# Adds an ensemble comprised of random projections
def add_ensemble_random(name, meta_name, meta, cross_val, X_train, Y_train, X_test, Y_test):
	# Establish and reset variables
	acc_score_cv = None
	acc_score = None
	time_ = None
	ensemble_cv = SuperLearner(scorer=accuracy_score, random_state=seed)
	ensemble = SuperLearner(scorer=accuracy_score, random_state=seed)
	models = []
	for j in range(0,30):
		#try out a new classifier
		pipeline1 = Pipeline([
			('rte', RandomTreesEmbedding(n_estimators=random.randint(50,150),max_depth=random.randint(1,200),random_state=random.randint(1,5000))),
			('rfc', RandomForestClassifier(n_estimators=random.randint(50,150),max_features=random.randint(1,20),max_depth=random.randint(1,200),random_state=random.randint(1,5000)))
		])
		models.append(pipeline1)

	if (cross_val == True):
		ensemble_cv.add(models)
		ensemble.add(models)

		# Attach the final meta estimator
		ensemble_cv.add_meta(meta)
		ensemble.add_meta(meta)

		ensemble_cv.fit(X_train[:1600], Y_train[:1600])
		preds_cv = ensemble_cv.predict(X_test[:1600])
		acc_score_cv = accuracy_score(preds_cv, Y_test[:1600])

		start = time.time()
		ensemble.fit(X_train, Y_train)
		preds = ensemble.predict(X_test)
		acc_score = accuracy_score(preds, Y_test)
		end = time.time()
		time_ = end - start
		if(acc_score > 0.83):
			if (len(listToSave) == 5):
				for x in listToSave:
					if x[1] < acc_score:
						x = [ensemble, acc_score]
						break;
			else:
				listToSave.append([ensemble, acc_score])
		listToSave.sort(key=itemgetter(1))

		return {"Ensemble_Elements": name, "Meta_Classifier": meta_name, "Accuracy_Score_CV": acc_score_cv, "Accuracy_Score": acc_score, "Runtime": time_}
	else:
		ensemble.add(models)
		# Attach the final meta estimator
		ensemble.add_meta(meta)

		start = time.time()
		ensemble.fit(X_train, Y_train)
		preds = ensemble.predict(X_test)
		acc_score = accuracy_score(preds, Y_test)
		end = time.time()
		time_ = end - start
		if(acc_score > 0.83):
			if (len(listToSave) == 5):
				for x in listToSave:
					if x[1] < acc_score:
						x = [ensemble, acc_score]
						break;
			else:
				listToSave.append([ensemble, acc_score])
		listToSave.sort(key=itemgetter(1))

		return {"Ensemble_Elements": name, "Meta_Classifier": meta_name, "Accuracy_Score_CV": "N/A", "Accuracy_Score": acc_score, "Runtime": time_}

# Adds an ensemble composed of a collection of models of different types
def ensemble_collection(elements, meta_name, meta, cross_val, X_train, Y_train, X_test, Y_test):
	# Establish and reset variables
	acc_score_cv = None
	acc_score = None
	time_ = None
	ensemble_cv = SuperLearner(scorer=accuracy_score, random_state=seed)
	ensemble = SuperLearner(scorer=accuracy_score, random_state=seed)
	models = []

	for element in elements:
		#try out a new classifier
		pipeline1 = Pipeline([
			(element[0], element[1])
		])
		models.append(pipeline1)

	if (cross_val == True):
		ensemble_cv.add(models)
		ensemble.add(models)

		# Attach the final meta estimator
		ensemble_cv.add_meta(meta)
		ensemble.add_meta(meta)

		ensemble_cv.fit(X_train[:1600], Y_train[:1600])
		preds_cv = ensemble_cv.predict(X_test[:1600])
		acc_score_cv = accuracy_score(preds_cv, Y_test[:1600])

		start = time.time()
		ensemble.fit(X_train, Y_train)
		preds = ensemble.predict(X_test)
		acc_score = accuracy_score(preds, Y_test)
		end = time.time()
		time_ = end - start
		if(acc_score > 0.83):
			if (len(listToSave) == 5):
				for x in listToSave:
					if x[1] < acc_score:
						x = [ensemble, acc_score]
						break;
			else:
				listToSave.append([ensemble, acc_score])
		listToSave.sort(key=itemgetter(1))

		return {"Ensemble_Elements": elements, "Meta_Classifier": meta_name, "Accuracy_Score_CV": acc_score_cv, "Accuracy_Score": acc_score, "Runtime": time_}
	else:
		ensemble.add(models)
		# Attach the final meta estimator
		ensemble.add_meta(meta)

		start = time.time()
		ensemble.fit(X_train, Y_train)
		preds = ensemble.predict(X_test)
		acc_score = accuracy_score(preds, Y_test)
		end = time.time()
		time_ = end - start
		if(acc_score > 0.83):
			if (len(listToSave) == 5):
				for x in listToSave:
					if x[1] < acc_score:
						x = [ensemble, acc_score]
						break;
			else:
				listToSave.append([ensemble, acc_score])
		listToSave.sort(key=itemgetter(1))

		return {"Ensemble_Elements": [elements[0][0],elements[1][0],elements[2][0]], "Meta_Classifier": meta_name, "Accuracy_Score_CV": "N/A", "Accuracy_Score": acc_score, "Runtime": time_}


# Adds old ensembles that performed well on previous datasets
def ensemble_saved():
	# Establish and reset variables
	acc_score_cv = None
	acc_score = None
	time_ = None

	#open from previous library... which is currently too large
	for m in listToSave:
		if (cross_val == True):

			m.fit(X_train[:1600], Y_train[:1600])
			preds_cv = m.predict(X_test[:1600])
			acc_score_cv = accuracy_score(preds_cv, Y_test[:1600])

			start = time.time()
			m.fit(X_train, Y_train)
			preds = m.predict(X_test)
			acc_score = accuracy_score(preds, Y_test)
			end = time.time()
			time_ = end - start

			return {"Ensemble_Elements": name, "Meta_Classifier": "svc", "Accuracy_Score_CV": acc_score_cv, "Accuracy_Score": acc_score, "Runtime": time_}
		else:
			start = time.time()
			m.fit(X_train, Y_train)
			preds = m.predict(X_test)
			acc_score = accuracy_score(preds, Y_test)
			end = time.time()
			time_ = end - start

			return {"Ensemble_Elements": name, "Meta_Classifier": "svc", "Accuracy_Score_CV": "N/A", "Accuracy_Score": acc_score, "Runtime": time_}

# Runs the program... add test datasets to this portion
def main():
	#read in data and parse
	files = ['obtrain.csv','obtest.csv']
	train_df = pd.read_csv(files[0], header=None)
	test_df = pd.read_csv(files[1], header=None)
	combine = [train_df, test_df]
	file_output = "output.txt"

	#map classifier as binary
	for dataset in combine:
			dataset[559] = dataset[559].map({1.0: 1, -1.0: 0}).astype(int)

	#separate models
	X_train = train_df.drop(559, axis=1)
	Y_train = train_df[559]
	X_test = test_df.drop(559, axis=1)
	Y_test = test_df[559]

	#feature selection (currently only works on datasets that do not have named index fields)
	selector = SelectKBest(f_classif, k=20)
	X_train = selector.fit_transform(X_train, Y_train)
	X_test = X_test[selector.get_support(indices=True)]

	#print("------Feature Selection Complete------")

	output = {} 

	# output['rfc'] = add_ensemble_same('rfc', RandomForestClassifier(n_estimators=random.randint(50,150),max_features=random.randint(1,20),max_depth=random.randint(1,200),random_state=random.randint(1,5000)), 'svc', SVC(), False, X_train, Y_train, X_test, Y_test)
	output['lr'] = add_ensemble_same('lr', LogisticRegression(random_state=random.randint(1,5000)), 'svc', SVC(), False, X_train, Y_train, X_test, Y_test)
	output['etc'] = add_ensemble_same('etc', ExtraTreeClassifier(max_features=random.randint(1,20),max_depth=random.randint(1,200),random_state=random.randint(1,5000)), 'svc', SVC(), False, X_train, Y_train, X_test, Y_test)
	output['svc'] = add_ensemble_same('svc', SVC(random_state=random.randint(1,5000),degree=random.randint(1,5000)), 'svc', SVC(), False, X_train, Y_train, X_test, Y_test)
	output['knc'] = add_ensemble_same('knc', KNeighborsClassifier(n_neighbors=random.randint(1,20),leaf_size=random.randint(10,100)), 'svc', SVC(), False, X_train, Y_train, X_test, Y_test)
	output['dtc'] = add_ensemble_same('dtc', DecisionTreeClassifier(max_depth=random.randint(1,200),max_features=random.randint(1,20),random_state=random.randint(1,5000)), 'svc', SVC(), False, X_train, Y_train, X_test, Y_test)
	# output['rp1'] = add_ensemble_random('rp1', 'svc', SVC(), False, X_train, Y_train, X_test, Y_test)
	output['combo'] = ensemble_collection(
			[['rfc', RandomForestClassifier(n_estimators=random.randint(50,150),max_features=random.randint(1,20),max_depth=random.randint(1,200),random_state=random.randint(1,5000))],
			['lr', LogisticRegression(random_state=random.randint(1,5000))],
			['knc', KNeighborsClassifier(n_neighbors=random.randint(1,20),leaf_size=random.randint(10,100))],
			['dtc', DecisionTreeClassifier(max_depth=random.randint(1,200),max_features=random.randint(1,20),random_state=random.randint(1,5000))]],
			'svc', SVC(), False, X_train, Y_train, X_test, Y_test)

	t = Texttable()
	t.add_row(['Dataset', 'Ensemble Components', 'Meta Classifier', 'Accuracy Score', 'Accuracy Score (cross-val)', 'Runtime'])
	for key, value in output.iteritems():
		t.add_row([key, output[key]["Ensemble_Elements"], output[key]["Meta_Classifier"], output[key]["Accuracy_Score"], output[key]["Accuracy_Score_CV"], output[key]["Runtime"]])
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
"""print("Savings {} high performing models into library (acc_score > 85%)".format(len(listToSave)))
with open('library.pkl', 'wb') as f:
	pickle.dump(listToSave, f)"""