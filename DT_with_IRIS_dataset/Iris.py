from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.tree import export_graphviz

iris = load_iris()
x = iris.data
y=iris.target

tree_clf = DecisionTreeClassifier()
model = tree_clf.fit(x,y)

dot_data = export_graphviz(tree_clf,out_file=None,feature_names=iris.feature_names,class_names=iris.target_names,filled=True,rounded=True,special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("iris")

prob = tree_clf.predict_proba([[7,3.3,4.5,1.5]])

print(prob)