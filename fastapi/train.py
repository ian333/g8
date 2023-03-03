from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
from lazypredict.Supervised import LazyClassifier


digits= datasets.load_digits()

X_d=digits.data
y_d=digits.target


X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(X_d, y_d, test_size=0.2, random_state=42)

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf_d = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
#clf_d.fit(X_d_train,y_d_train)
models,predictions = clf_d.fit(X_d_train, X_d_test, y_d_train, y_d_test)


clf = SVC(kernel='linear', C=1, gamma='auto')
clf.fit(X_train, y_train)


accuracy = clf.score(X_test, y_test)
print(f"Accuracy Iris: {accuracy}")

print(models)


#joblib.dump(clf, "models/digits_svm_model.pkl")



joblib.dump(clf, "models/iris_svm_model.pkl")

