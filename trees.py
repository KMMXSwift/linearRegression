import graphviz
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

X, y = load_iris(return_X_y=True)
print(len(X))

X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8)
print(len(X_train), len(Y_train), len(X_test), len(Y_test))

tree = DecisionTreeClassifier(criterion="entropy")
tree = tree.fit(X_train, Y_train)

prediction = tree.predict(X_test)
print(prediction)
print(Y_test)

score = tree.score(X_test, Y_test)
print(score)

data = export_graphviz(tree, out_file=None,
                       feature_names=["sepal length", "sepal width", "petal length", "petal width"],
                       class_names=["setosa", "versicolor", "virginica"])
graph = graphviz.Source(data)
graph.render("tree")

forest = RandomForestClassifier(n_estimators=100, criterion="gini")
forest.fit(X_train, Y_train)

prediction = forest.predict(X_test)
print(prediction)

score = forest.score(X_test, Y_test)
print(score)

gbt = AdaBoostClassifier(n_estimators=100)
gbt.fit(X_train, Y_train)

prediction = gbt.predict(X_test)
print(prediction)

score = gbt.score(X_test, Y_test)
print(score)
