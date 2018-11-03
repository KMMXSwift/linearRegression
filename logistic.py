from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)

classifier = LogisticRegression(solver="saga", multi_class="multinomial").fit(X, y)

test_x = X[:8]
print(classifier.predict(test_x))
print(classifier.score(X, y))
