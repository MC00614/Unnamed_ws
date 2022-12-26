from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

iris = load_iris()
iris_data = iris.data
iris_label = iris.target


iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label']=iris.target

X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)


dt_clf = DecisionTreeClassifier(random_state=11)
dt_clf.fit(X_train, y_train)


pred = dt_clf.predict(X_test)
print(accuracy_score(y_test, pred))
