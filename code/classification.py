import time
from sklearn.model_selection import KFold
from sklearn import model_selection
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from loadData import load_data
ld = load_data()


target_names = ['Playoff','Playoff']

def run_SVM(X_train,X_test,y_train,y_test):
    # Training the SVM model using X_train and Y_train
    start_time = time.time()
    svm = SVC(decision_function_shape='ovr',gamma='scale')
    svm.fit(X_train, y_train)
    print("---Training Time %s seconds ---" % (time.time() - start_time))
    # Classification of X_test using the SVM model
    start_time = time.time()
    predictions = svm.predict(X_test)
    # Performance measure
    # use the classification report in order to extract the average F1 measure
    print(classification_report(y_test, predictions,target_names=target_names))
    # displaying the classification performances through the confusion matrix as well.
    print(confusion_matrix(y_test, predictions))
    print("--- Testing Time %s seconds ---" % (time.time() - start_time))
    
    return 0

def run_GNB(X_train,X_test,y_train,y_test):
    start_time = time.time()
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    print("---Training Time %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    predictions = gnb.predict(X_test)

    # Performance measure
    # use the classification report in order to extract the average F1 measure
    print(classification_report(y_test, predictions,target_names=target_names))
    # displaying the classification performances through the confusion matrix as well.
    print(confusion_matrix(y_test, predictions))
    print("--- Testing Time %s seconds ---" % (time.time() - start_time))
    return 0

###test-start
data = np.array(ld.collect()).astype(np.float64)
X = data[:,2:]
y = data[:,1]

X_train = X[0:774]
X_test = X[775:]
y_train = y[0:774]
y_test = y[775:]
print("starting 70:30 traing:testing")
print("############## Algorithm 1: Support Vector Machines #################")
run_SVM(X_train, X_test, y_train, y_test)

print("############## Algorithm 2: Gaussian Naive Bayes #################")
run_GNB(X_train, X_test, y_train, y_test)

print("end of 70:30 traing:testing")
###test-end

ns = 5
print("############## model-0: all columns #################")
data = np.array(ld.collect()).astype(np.float64)
X = data[:,2:]
y = data[:,1]
# KFold intro: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
kf = KFold(n_splits=ns, random_state=None, shuffle=False)
count=0
for train_index, test_index in kf.split(X):
    count+=1
    print("################### K-FOLD Round "+str(count)+" ##########################")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print("############## Algorithm 1: Support Vector Machines #################")
    run_SVM(X_train, X_test, y_train, y_test)

    print("############## Algorithm 2: Gaussian Naive Bayes #################")
    run_GNB(X_train, X_test, y_train, y_test)

print("############## model-1: removed 3Points_Per_minute and 2Points_Per_minute #################")
data_1 = ld.select("id","Playoff","Points_Per_minute","FThrow_Per_minute","Rebound_Per_minute","Assists_Per_minute","Steals_Per_minute","Blocks_Per_minute","TurnOvers_Per_minute").collect()
data = np.array(data_1).astype(np.float64)
X_1 = data[:,2:]
y_1 = data[:,1]

# KFold intro: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
kf = KFold(n_splits=ns, random_state=None, shuffle=False)
count=0
for train_index, test_index in kf.split(X_1):
    count+=1
    print("################### K-FOLD Round "+str(count)+" ##########################")
    X_train, X_test = X_1[train_index], X_1[test_index]
    y_train, y_test = y_1[train_index], y_1[test_index]

    print("############## Algorithm 1: Support Vector Machines #################")
    run_SVM(X_train, X_test, y_train, y_test)

    print("############## Algorithm 2: Gaussian Naive Bayes #################")
    run_GNB(X_train, X_test, y_train, y_test)

print("############## model-2: removed Points_Per_minute and 3Points_Per_minute #################")
data_2 = ld.select("id","Playoff","2Points_Per_minute","FThrow_Per_minute","Rebound_Per_minute","Assists_Per_minute","Steals_Per_minute","Blocks_Per_minute","TurnOvers_Per_minute").collect()
data = np.array(data_2).astype(np.float64)
X_2 = data[:,2:]
y_2 = data[:,1]

# KFold intro: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
kf = KFold(n_splits=ns, random_state=None, shuffle=False)
count=0
for train_index, test_index in kf.split(X_2):
    count+=1
    print("################### K-FOLD Round "+str(count)+" ##########################")
    X_train, X_test = X_2[train_index], X_2[test_index]
    y_train, y_test = y_2[train_index], y_2[test_index]

    print("############## Algorithm 1: Support Vector Machines #################")
    run_SVM(X_train, X_test, y_train, y_test)

    print("############## Algorithm 2: Gaussian Naive Bayes #################")
    run_GNB(X_train, X_test, y_train, y_test)

print("############## model-3: removed Points_Per_minute and 2Points_Per_minute #################")
data_3 = ld.select("id","Playoff","3Points_Per_minute","FThrow_Per_minute","Rebound_Per_minute","Assists_Per_minute","Steals_Per_minute","Blocks_Per_minute","TurnOvers_Per_minute").collect()
data = np.array(data_3).astype(np.float64)
X_3 = data[:,2:]
y_3 = data[:,1]

# KFold intro: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
kf = KFold(n_splits=ns, random_state=None, shuffle=False)
count=0
for train_index, test_index in kf.split(X_3):
    count+=1
    print("################### K-FOLD Round "+str(count)+" ##########################")
    X_train, X_test = X_3[train_index], X_3[test_index]
    y_train, y_test = y_3[train_index], y_3[test_index]

    print("############## Algorithm 1: Support Vector Machines #################")
    run_SVM(X_train, X_test, y_train, y_test)

    print("############## Algorithm 2: Gaussian Naive Bayes #################")
    run_GNB(X_train, X_test, y_train, y_test)
