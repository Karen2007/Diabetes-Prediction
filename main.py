import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("diabetes.csv")

# Adding some new features
df["Age*BMI"] = df["Age"] * df["BMI"]
df["Age*BloodPressure"] = df["Age"] * df["BloodPressure"]
df["Age^2"] = df["Age"] * df["Age"]
df["BMI^2"] = df["BMI"] * df["BMI"]
df["BloodPressure^2"] = df["BloodPressure"] * df["BloodPressure"]
df["HighBMI"] = df["BMI"] >= df["BMI"].quantile(0.75)
# df["OldAge"] = df["Age"] >= 60
df["HighInsulin"] = df["Insulin"] >= df["Insulin"].quantile(0.75)
df["HighGlucose"] = df["Glucose"] >= df["Glucose"].quantile(0.75)

corr = df.corr()
print(corr["Outcome"].sort_values(ascending=False))

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

average_acc = 0
average_precision = 0
average_recall = 0
average_f1 = 0

estimator = LogisticRegression(random_state=42)

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "class_weight": [None, "balanced"],
    "solver": ["liblinear", "saga"],
    "max_iter": [1000, 5000, 10000],
}

grid = GridSearchCV(estimator=estimator, param_grid=param_grid, verbose=3, scoring="f1")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

cols_to_impute = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
cols_to_scale = ["Glucose", "Age*BMI", "BMI^2", "BMI", "Age", "Pregnancies", "Age*BloodPressure", "BloodPressure",
                 "Age^2", "Insulin",
                 "DiabetesPedigreeFunction", "BloodPressure^2", "SkinThickness"]

imputer = SimpleImputer(missing_values=0, strategy="median")
scaler = StandardScaler()

X_train[cols_to_impute] = imputer.fit_transform(X_train[cols_to_impute])
X_test[cols_to_impute] = imputer.transform(X_test[cols_to_impute])

X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

grid.fit(X_train, y_train)

grid.predict(X_test)
print(grid.best_estimator_)

for i in range(1, 41):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=i)

    cols_to_impute = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    cols_to_scale = ["Glucose", "Age*BMI", "BMI^2", "BMI", "Age", "Pregnancies", "Age*BloodPressure", "BloodPressure", "Age^2", "Insulin",
                     "DiabetesPedigreeFunction", "BloodPressure^2", "SkinThickness"]

    imputer = SimpleImputer(missing_values=0, strategy="median")
    scaler = StandardScaler()

    X_train[cols_to_impute] = imputer.fit_transform(X_train[cols_to_impute])
    X_test[cols_to_impute] = imputer.transform(X_test[cols_to_impute])

    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    grid.best_estimator_.fit(X_train, y_train)

    grid.best_estimator_.predict(X_test)

    print(f"Random state {i}:")
    print(f"Accuracy: {grid.best_estimator_.score(X_test, y_test)}")
    print(f"Precision: {precision_score(y_test, grid.best_estimator_.predict(X_test))}")
    print(f"Recall: {recall_score(y_test, grid.best_estimator_.predict(X_test))}")
    print(f"F1 Score: {f1_score(y_test, grid.best_estimator_.predict(X_test))}")

    average_acc += grid.best_estimator_.score(X_test, y_test)
    average_precision += precision_score(y_test, grid.best_estimator_.predict(X_test))
    average_f1 += f1_score(y_test, grid.best_estimator_.predict(X_test))
    average_recall += recall_score(y_test, grid.best_estimator_.predict(X_test))

print("\nAverages:")
print(f"Average accuracy: {average_acc / 40:.3f}")
print(f"Average precision: {average_precision / 40:.3f}")
print(f"Average recall: {average_recall / 40:.3f}")
print(f"Average F1 Score: {average_f1 / 40:.3f}")