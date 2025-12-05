import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import xlabel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

titanic_data = pd.read_csv('titanic.csv')
titanic_data["sex"] = titanic_data["sex"].map({"male":0, "female": 1})
titanic_data["age"] = titanic_data["age"].fillna(titanic_data["age"].median())
titanic_data["family_size"] = titanic_data["sibsp"] + titanic_data["parch"] + 1
titanic_data["is_alone"] = (titanic_data["family_size"] == 1).astype(int)
titanic_data["age"] = titanic_data["age"].fillna(titanic_data["age"].median())
titanic_data["fare"] = titanic_data["fare"].fillna(titanic_data["fare"].median())
y = titanic_data["survived"]
x = titanic_data.drop(columns=["name", "survived"])
print(titanic_data.isna())
print(x.head())
print(x.tail())
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, test_size=0.4, shuffle=True)
test = x_train.shape
scaler = StandardScaler()
x_scaled_train = scaler.fit_transform(x_train.values)
x_scaled_test = scaler.transform(x_test.values)
model = LogisticRegression(max_iter=1000)
model.fit(x_scaled_train, y_train.values)
accuracy = model.score(x_scaled_test, y_test)
scores = cross_val_score(model, x, y, cv = 5)
preds = model.predict(x_scaled_test)
num_survived = sum(preds)
print(f"Model Accuracy on Test Set: {accuracy*100:.2f}%")
print("Number of predicted survivors: ", num_survived)
print("Actual number of survivors: ", sum(y_test))
print("Cross-validation scores:", scores)
print("Mean CV score:", scores.mean())
titanic_data.dropna(subset=['age', 'fare'], inplace=True)
