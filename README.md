# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Start
Steps

1.Collect student data including grades and placement status.

2.Clean and split the data for training and testing.

3.Train the logistic regression model using the training data.

4.Evaluate the model's performance using testing data.

5.Use the trained model to predict placement status for new students.

Stop
## Program:
```py
'''
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Vikamuhan reddy.N
RegisterNumber:  212223240181
'''
import pandas as pd
data = pd.read_csv("D:/introduction to ML/jupyter notebooks/sample/Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data.head()
data.isnull()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")# library for large linear classificiation
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85,]])

```

## Output:
![image](https://github.com/vikamuhan-reddy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144928933/5d701a7a-e58e-4d18-a468-98bdb2b7ad91)
![image](https://github.com/vikamuhan-reddy/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144928933/cd46c18d-ca35-48c0-8324-902f8694af4a)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
