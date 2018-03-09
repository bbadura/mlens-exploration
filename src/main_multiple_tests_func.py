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

def add_ensemble_same(name, element):
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

	ensemble_cv.add(models)
	ensemble.add(models)

	# Attach the final meta estimator
	ensemble_cv.add_meta(SVC())
	ensemble.add_meta(SVC())

	ensemble_cv.fit(X_train[i][:1600], Y_train[i][:1600])
	preds_cv = ensemble_cv.predict(X_test[i][:1600])
	acc_score_cv = accuracy_score(preds_cv, Y_test[i][:1600])

	start = time.time()
	ensemble.fit(X_train[i], Y_train[i])
	preds = ensemble.predict(X_test[i])
	acc_score = accuracy_score(preds, Y_test[i])
	end = time.time()
	time_ = end - start

	return {"Ensemble_Elements": name, "Meta_Classifier": "SVC", "Accuracy_Score_CV": acc_score_cv, "Accuracy_Score": acc_score, "Runtime": time_}


#read in data and parse
files = [['obtrain.csv','obtest.csv']]
train_df = []
test_df = []
combine = []
for i, file in enumerate(files):
	train_df.append(pd.read_csv(file[0], header=None))
	test_df.append(pd.read_csv(file[1], header=None))
	combine.append([train_df[i], test_df[i]])

#map classifier as binary
for element in combine:
	for dataset in element:
		dataset[559] = dataset[559].map({1.0: 1, -1.0: 0}).astype(int)

#separate models
X_train = []
Y_train = []
X_test = []
Y_test = []

for i in range(len(combine)):
	X_train.append(train_df[i].drop(559, axis=1))
	Y_train.append(train_df[i][559])
	X_test.append(test_df[i].drop(559, axis=1))
	Y_test.append(test_df[i][559])

seed = 9880

#feature selection
selector = SelectKBest(f_classif, k=20)
for i in range(len(combine)):
	X_train[i] = selector.fit_transform(X_train[i], Y_train[i])
	X_test[i] = X_test[i][selector.get_support(indices=True)]

print("------Feature Selection Complete------")

output = {}

output['rfc'] = add_ensemble_same('rfc', RandomForestClassifier(n_estimators=random.randint(50,150),max_features=random.randint(1,20),max_depth=random.randint(1,200),random_state=random.randint(1,5000)))
output['lr'] = add_ensemble_same('lr', LogisticRegression(random_state=random.randint(1,5000)))
output['etc'] = add_ensemble_same('etc', ExtraTreeClassifier(max_features=random.randint(1,20),max_depth=random.randint(1,200),random_state=random.randint(1,5000)))
output['svc'] = add_ensemble_same('svc', SVC(random_state=random.randint(1,5000), degree=random.randint(1,5000)))

t = Texttable()
t.add_row(['Dataset', 'Ensemble Components', 'Meta Classifier', 'Accuracy Score', 'Accuracy Score (cross-val)', 'Runtime'])
for key, value in output.iteritems():
	t.add_row([key, output[key]["Ensemble_Elements"], output[key]["Meta_Classifier"], output[key]["Accuracy_Score"], output[key]["Accuracy_Score_CV"], output[key]["Runtime"]])
print(t.draw())
