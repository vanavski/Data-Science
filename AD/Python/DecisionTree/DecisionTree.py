import pydot
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

df=pd.read_excel("ONLINEADS.xlsx",sheet_name = 0)
x=df[['Age', 'Gender', 'Interest', 'VisitTime', 'City','Device','OS','VisitPage','AdsTool','VisitNumber']]
y=df['Registration']
y=y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

print(df.shape)
print(df.info())

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

feature_names = ['Age', 'Gender', 'Interest', 'VisitTime', 'City','Device','OS','VisitPage','AdsTool','VisitNumber']

export_graphviz(tree, out_file='D:/AD/Python/DecisionTree/onlineads.dot',
rounded = True, proportion = False, feature_names=feature_names,
precision = 2, filled = True)

(graph, ) = pydot.graph_from_dot_file('D:/AD/Python/DecisionTree/onlineads.dot')

graph.write_png('tree.png')