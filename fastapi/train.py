from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import ExtraTreesClassifier


digits=datasets.load_digits()
X_d=digits.data
y_d=digits.target

X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(X_d, y_d,test_size=.2)
clf_d=ExtraTreesClassifier(n_estimators=10)
clf_d.fit(X_d_train,y_d_train)



iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



clf = SVC(kernel='linear', C=1, gamma='auto')
clf.fit(X_train, y_train)


accuracy_digits = clf_d.score(X_d_test, y_d_test)
print(f"Accuracy de digitos: {accuracy_digits}")

accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

joblib.dump(clf, "models/iris_svm_model.pkl")

joblib.dump(clf_d, "models/digits_ExtratreeC_model.pkl")

