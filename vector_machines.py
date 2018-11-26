from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
print(len(X))

X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8)
print(len(X_train), len(Y_train), len(X_test), len(Y_test))

svm = SVC()
svm = svm.fit(X_train, Y_train)

prediction = svm.predict(X_test)
print(prediction)
print(Y_test)

score = svm.score(X_test, Y_test)
print(score)
