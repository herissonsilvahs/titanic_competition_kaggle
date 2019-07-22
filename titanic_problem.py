import pandas as pd
import numpy as np
import warnings
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import math

warnings.filterwarnings("ignore")
titanic_train = pd.read_csv('./train.csv')
titanic_test = pd.read_csv('./test.csv')
train_predict_labels = ['Age', 'Sex','Pclass','Fare', 'Embarked', 'SibSp', 'Parch']

titanic_train['Age'].fillna(value=int(math.floor(titanic_train['Age'].mean(skipna=True))), inplace=True)
titanic_test['Age'].fillna(value=int(math.floor(titanic_test['Age'].mean(skipna=True))), inplace=True)
titanic_train['Embarked'].fillna(value='S', inplace=True)
titanic_test['Fare'].fillna(value=math.floor(titanic_test['Fare'].mean(skipna=True)), inplace=True)

x_train = titanic_train[train_predict_labels]
x_test = titanic_test[train_predict_labels]
y_train = titanic_train['Survived']

x_train['Sex'] = pd.get_dummies(x_train['Sex'], drop_first=True)
x_test['Sex'] = pd.get_dummies(x_test['Sex'], drop_first=True)

x_train_embarked = pd.get_dummies(x_train['Embarked'])
x_test_embarked = pd.get_dummies(x_test['Embarked'])

x_train = pd.concat([x_train, x_train_embarked], axis=1)
x_test = pd.concat([x_test, x_test_embarked], axis=1)

x_train.drop(['Embarked'], axis=1, inplace=True)
x_test.drop(['Embarked'], axis=1, inplace=True)

# min_max_scaler = MinMaxScaler()
# min_max_scaler.fit(x_train)
# x_train = min_max_scaler.transform(x_train)
# x_test = min_max_scaler.transform(x_test)

# randomForestModel = RandomForestClassifier()
# randomForestModel.fit(x_train, y_train)

# param_grid = {
#   'n_estimators' : [200, 500, 1000, 500, 1000, 1000, 500],
#   'max_features' : ['auto', 'auto', 'auto', 'sqrt', 'log2', 'sqrt', 'log2']
# }

# gscv = GridSearchCV(estimator=randomForestModel, param_grid=param_grid, verbose=True)
# gscv.fit(x_train, y_train)

# print(gscv.best_score_) # 0.8069584736251403
# print(gscv.best_params_) # {'max_features': 'auto', 'n_estimators': 500}


# score = randomForestModel.score(x_train, y_train)
# predicts = randomForestModel.predict(x_test)

# logisticRegressionModel = LogisticRegression()

# param_grid = {
#   'penalty': ['l1', 'l2'],
#   'C': np.logspace(-4, 4, 20)
# }

# gscv = GridSearchCV(estimator=logisticRegressionModel, param_grid=param_grid, verbose=True)
# gscv.fit(x_train, y_train)

# print(gscv.best_score_) # 0.7934904601571269
# print(gscv.best_params_) # {'C': 0.23357214690901212, 'penalty': 'l1'}

logisticRegressionModel = LogisticRegression(penalty='l1', C=0.23357214690901212)
logisticRegressionModel.fit(x_train, y_train)

score_lg = logisticRegressionModel.score(x_train, y_train)
predicts_lg = logisticRegressionModel.predict(x_test)

data = {'PassengerId': np.array(titanic_test['PassengerId']), 'Survived': predicts_lg}
result_predict_data = pd.DataFrame(data, columns=['PassengerId', 'Survived'])
result_predict_data.to_csv('result_predict_data.csv', index=None)

# print("Score Random Forest: "+str(score))
# print(predicts)

# print(80*"#")

print("Score Logistic Regression: "+str(score_lg))
# print(predicts_lg)