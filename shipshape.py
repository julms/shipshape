#!/anaconda/bin/python

import sys
import argparse
from pandas import read_csv, DataFrame, concat
from time import time,strftime,localtime
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-base",required=True)
parser.add_argument("-new",required=True)
start = time()
args = parser.parse_args()

def local_time():
	return strftime("%d %b %Y %I:%M:%S %p %Z",localtime())

print "Starting operation at time",local_time()

print "Reading base data... \n"

base_data = read_csv(args.base)
base_data_numeric = base_data._get_numeric_data()

print "Base data read successfully. Filtered only numeric columns.\n",
print "Base data contains", len(base_data_numeric), "rows and", len(base_data_numeric.columns), "columns.\n"

#print "Removing the Primary Key variable for the base data, if applicable..."
#base_data = base_data.drop(args.key)

sys.stdout.flush()
print "Reading new data... \n"

new_data = read_csv(args.new)

new_data_numeric = new_data._get_numeric_data()

print "New data read successfully. Filtered only numeric columns\n",
print "New data contains", len(new_data_numeric), "rows and", len(new_data_numeric.columns), "columns.\n"


#print "Removing the Primary Key variable for the test data, if applicable..."
#test_data = test_data.drop(args.key)

sys.stdout.flush()
#print local_time(), "Summarizing data... "

base_mean = base_data_numeric.mean()

new_mean = new_data_numeric.mean()

percentage_mean_diff = [abs((y-x)/float(x)*100.0) for x,y in zip(base_mean,new_mean)]

#print percentage_mean_diff
print "TEST 1: Compare the deviation in means between the two datasets"


meanlist = []

for i in percentage_mean_diff:
	if i > 5:
		meanlist.append(i)

if len(meanlist) > (2/float(3))*len(percentage_mean_diff):
	print "SUBSTANTIAL DIFFERENCE IN MEANS: CODE RED\n"

elif len(meanlist) > (1/float(3))*len(percentage_mean_diff):
	print "MODERATE DIFFERENCE IN MEANS: CODE YELLOW\n"

else:
	print "SLIM DIFFERENCE IN MEANS: CODE GREEN\n"

print "There are", len(meanlist), "columns of", len(percentage_mean_diff), "total with means that differ by > 5%.\n"

sys.stdout.flush()

base_var = base_data_numeric.var()

new_var = new_data_numeric.var()

percentage_var_diff = [abs(y-x)/float(x+1) for x,y in zip(base_var, new_var)]


print "TEST 2: Compare the deviation in variances between the two datasets"
#print percentage_var_diff

varlist = []

for i in percentage_var_diff:
	if i > 5:
		varlist.append(i)

if len(varlist) > (2/float(3))*len(percentage_var_diff):
	print "SUBSTANTIAL DIFFERENCE IN VARIANCE: CODE RED\n"

elif len(varlist) > (1/float(3))*len(percentage_var_diff):
	print "MODERATE DIFFERENCE IN VARIANCE: CODE YELLOW\n"

else:
	print "SLIM DIFFERENCE IN VARIANCE: CODE GREEN\n"

print "There are", len(varlist), "columns of", len(percentage_var_diff), "total with variance that differ by > 5%.\n"


sys.stdout.flush()

base_data_v1 = base_data_numeric.copy()
new_data_v1 = new_data_numeric.copy()

base_data_v1['target'] = 0
new_data_v1['target'] = 1



base_data_v2 = base_data_v1.sample(frac=0.1, replace=False)
new_data_v2 = new_data_v1.sample(frac=0.1,replace=False)

combined_data = concat([base_data_v2,new_data_v2],axis=0)


combined_data = combined_data.fillna(-999999)

target_variable = combined_data['target']

independent_variables = combined_data.drop("target",axis=1)


x_train, x_test, y_train, y_test = train_test_split(independent_variables, target_variable, test_size=0.3, random_state=10)
clf = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.05, max_depth = 3, verbose=True)

print "Test 3: Create a new variable called \"target\""
print "This variable is given the value 0 for the base data, and 1 for the test data"
print "Take 10%  of each of data set and concatenate into one file"
print "Now, form a random 70:30 split between your train and test data to run a Gradient Boosted Tree algorithm"
print "The model will determine variables that are most indicative of each data set"
print "These variables could potentially warrant further inquiry.\n"

print clf
print "\n"

sys.stdout.flush()

clf.fit(x_train,y_train)



worst_variables = DataFrame(list(base_data_numeric.columns),columns = ['variable_name'])
worst_variables['impact'] = clf.feature_importances_


worst_variables = worst_variables[worst_variables['impact']>0.05]

worst_variables = worst_variables.sort_values('impact', ascending=False)

print "\n",len(worst_variables),"Variables most responsible for difference are\n"
print worst_variables.head(len(worst_variables))
end = time()

total_seconds = int(end-start)
hours = total_seconds/3600
minutes = (total_seconds-hours*60)/60
seconds = total_seconds - hours*3600 - minutes*60

print "Time taken to complete this exercise is",hours,"hour(s),",minutes,"minute(s) and",seconds,"second(s)"
sys.stdout.flush()
