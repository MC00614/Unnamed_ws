from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target



features = iris.data
label = iris.target

df_clf = DecisionTreeClassifier(random_state=156)

skfold = StratifiedKFold(n_splits=3)
cv_accuracy = []

for train_index, test_index, in skfold.split(features, label):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    df_clf.fit(X_train,y_train)
    pred = df_clf.predict(X_test)
    cv_accuracy.append(np.round(accuracy_score(y_test, pred),4))

print(np.round(np.mean(cv_accuracy),4))
