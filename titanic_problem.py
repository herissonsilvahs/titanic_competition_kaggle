import pandas as pd
import numpy as np
import warnings
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import math

warnings.filterwarnings("ignore")
titanic_train = pd.read_csv('./train.csv')
titanic_test = pd.read_csv('./test.csv')
train_predict_labels = ['Age', 'Sex','Pclass','Fare', 'Embarked', 'SibSp', 'Parch']

titanic_train['Age'].fillna(value=int(math.floor(titanic_train['Age'].mean(skipna=True))), inplace=True)
titanic_test['Age'].fillna(value=int(math.floor(titanic_test['Age'].mean(skipna=True))), inplace=True)
titanic_train['Embarked'].fillna(value='S', inplace=True)
titanic_test['Embarked'].fillna(value='S', inplace=True)
titanic_test['Fare'].fillna(value=math.floor(titanic_test['Fare'].mean(skipna=True)), inplace=True)

x_train = titanic_train[train_predict_labels]
x_test = titanic_test[train_predict_labels]
y_train = titanic_train['Survived']

labelEncoder = LabelEncoder()
labelEncoder2 = LabelEncoder()
labelEncoder3 = LabelEncoder()

labelEncoder.fit(x_train['Sex'].astype(str))
x_train['Sex'] = labelEncoder.transform(x_train['Sex'].astype(str))
x_test['Sex'] = labelEncoder.transform(x_test['Sex'].astype(str))
labelEncoder2.fit(x_train['Pclass'].astype(str))
x_train['Pclass'] = labelEncoder2.transform(x_train['Pclass'].astype(str))
x_test['Pclass'] = labelEncoder2.transform(x_test['Pclass'].astype(str))
labelEncoder3.fit(x_train['Embarked'].astype(str))
x_train['Embarked'] = labelEncoder3.transform(x_train['Embarked'].astype(str))
x_test['Embarked'] = labelEncoder3.transform(x_test['Embarked'].astype(str))

randomForestModel = RandomForestClassifier(n_estimators=500)
randomForestModel.fit(x_train, y_train)

score = randomForestModel.score(x_train, y_train)
predicts = randomForestModel.predict(x_test)

data = {'PassengerId': np.array(titanic_test['PassengerId']), 'Survived': predicts}
result_predict_data = pd.DataFrame(data, columns=['PassengerId', 'Survived'])
result_predict_data.to_csv('result_predict_data.csv', index=None)