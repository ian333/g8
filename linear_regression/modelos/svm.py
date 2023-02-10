import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import GaussianNB


iris = datasets.load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(X_train, y_train)


clf_bayes = GaussianNB()
clf_bayes.fit(X_train, y_train)


svm_accuracy = clf_svm.score(X_test, y_test)
bayes_accuracy = clf_bayes.score(X_test, y_test)

print(f"Accuracy of SVM: {svm_accuracy:.2f}")
print(f"Accuracy of Bayes: {bayes_accuracy:.2f}")


