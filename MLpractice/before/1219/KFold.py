from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()

# print(iris.keys())

features = iris.data
label = iris.target

df_clf = DecisionTreeClassifier(random_state=156)

kfold = KFold(n_splits=5)
cv_accuracy = []
print(features.shape)

# n_iter = 0

for train_index, test_index, in kfold.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    df_clf.fit(X_train,y_train)
    pred = df_clf.predict(X_test)
    cv_accuracy.append(np.round(accuracy_score(y_test, pred),4))

print(np.mean(cv_accuracy))
