from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib



iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



clf = SVC(kernel='linear', C=1, gamma='auto')
clf.fit(X_train, y_train)


accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

joblib.dump(clf, "models/iris_svm_model.pkl")

