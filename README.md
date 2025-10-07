# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Kishore R
RegisterNumber: 25011776 
*/
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("/content/Employee.csv")

# Basic info
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["left"].value_counts())

# Encode categorical column
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
print(data.head())

# Feature and target selection
x = data[[
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "salary"
]]
y = data["left"]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100
)

# Train Decision Tree
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)

# Predict
y_pred = dt.predict(x_test)

# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Example prediction
example = [[0.5, 0.8, 9, 260, 6, 0, 1, 2]]
pred = dt.predict(example)
print("Prediction for new employee:", "Left" if pred[0] == 1 else "Stayed")

# Plot decision tree
plt.figure(figsize=(12, 8))
plot_tree(
    dt,
    feature_names=x.columns,
    class_names=['Stayed', 'Left'],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree - Employee Churn Prediction")
plt.show()
```

## Output:
```
   satisfaction_level  last_evaluation  number_project  ...  promotion_last_5years  Departments   salary
0                0.38             0.53               2  ...                      0         sales     low
1                0.80             0.86               5  ...                      0         sales  medium
2                0.11             0.88               7  ...                      0         sales  medium
3                0.72             0.87               5  ...                      0         sales     low
4                0.37             0.52               2  ...                      0         sales     low

[5 rows x 10 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14999 entries, 0 to 14998
Data columns (total 10 columns):
 #   Column                 Non-Null Count  Dtype
---  ------                 --------------  -----
 0   satisfaction_level     14999 non-null  float64
 1   last_evaluation        14999 non-null  float64
 2   number_project         14999 non-null  int64
 3   average_montly_hours   14999 non-null  int64
 4   time_spend_company     14999 non-null  int64
 5   Work_accident          14999 non-null  int64
 6   left                   14999 non-null  int64
 7   promotion_last_5years  14999 non-null  int64
 8   Departments            14999 non-null  object
 9   salary                 14999 non-null  object
dtypes: float64(2), int64(6), object(2)
memory usage: 1.1+ MB
None
satisfaction_level       0
last_evaluation          0
number_project           0
average_montly_hours     0
time_spend_company       0
Work_accident            0
left                     0
promotion_last_5years    0
Departments              0
salary                   0
dtype: int64
left
0    11428
1     3571
Name: count, dtype: int64
   satisfaction_level  last_evaluation  number_project  ...  promotion_last_5years  Departments   salary
0                0.38             0.53               2  ...                      0         sales       1
1                0.80             0.86               5  ...                      0         sales       2
2                0.11             0.88               7  ...                      0         sales       2
3                0.72             0.87               5  ...                      0         sales       1
4                0.37             0.52               2  ...                      0         sales       1

[5 rows x 10 columns]
Accuracy: 0.9853333333333333
```

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/0eed9466-4f19-47f3-b837-032d54d4ec91" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
