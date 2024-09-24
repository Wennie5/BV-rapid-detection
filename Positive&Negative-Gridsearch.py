import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

df1 = pd.read_csv('D:/Desktop/BV/BV-SERS data/阴性阳性_norm.csv')
x1 = df1.values[:, 0:1161].astype(float)
y1 = df1.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(x1, y1, train_size=0.8, random_state=0)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

param = {'max_depth': range(1, 30),
         'max_features': range(1, 30),
         'criterion': ["entropy", "gini"]}

grid = GridSearchCV(DecisionTreeClassifier(), param_grid=param, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': [120, 130, 140, 150, 160],
              'max_depth': range(1, 10),
              'criterion': ['gini', 'entropy']}

grid = GridSearchCV(estimator=RandomForestClassifier(max_features='sqrt'), param_grid=parameters, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

# SVM
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1],
              'gamma': [0.0001, 0.001, 0.01, 0.1, 1]}

grid = GridSearchCV(SVC(kernel='sigmoid', probability=True), param_grid, cv=5)
grid.fit(X_train, y_train)

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': [50, 60, 70, 80, 90, 100],
              'learning_rate': [0.1, 1, 0.01, 0.001]}

grid = GridSearchCV(AdaBoostClassifier(), param_grid=parameters, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

# XGBoost
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': [50, 60, 70, 80, 90],
              'learning_rate': [0.1, 1, 0.01, 0.001]}

grid = GridSearchCV(XGBClassifier(), param_grid=parameters, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

# Gradient Boosting (GBoost)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': [120, 130, 140, 150, 160],
              'learning_rate': [0.1, 1, 0.01]}

grid = GridSearchCV(GradientBoostingClassifier(), param_grid=parameters, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

