#Importing basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport

#Importing the Dataset
diabetes = pd.read_csv(“diabetes.csv”)
dataset = diabetes

#EDA using Pandas Profiling
file = ProfileReport(dataset)
file.to_file(output_file=’output.html’)

from scipy.stats import shapiro
stat, p = shapiro(dataset['BloodPressure'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
#Statistics=0.819, p=0.000

#Classifying the Blood Pressure based on class
ax = sns.violinplot(x=”Outcome”, y=”BloodPressure”, data=dataset, palette=”muted”, split=True)

#Replacing the zero-values for Blood Pressure
df1 = dataset.loc[dataset['Outcome'] == 1]
df2 = dataset.loc[dataset['Outcome'] == 0]
df1 = df1.replace({'BloodPressure':0}, np.median(df1['BloodPressure']))
df2 = df2.replace({'BloodPressure':0}, np.median(df2['BloodPressure']))
dataframe = [df1, df2]
dataset = pd.concat(dataframe)

#Correlation
from scipy.stats import pearsonr
corr, _ = pearsonr(dataset[‘Age’], dataset[‘Pregnancies’])
print(‘Pearsons correlation: %.3f’ % corr)
#Pearsons correlation: 0.544

#Splitting the data into dependent and independent variables
Y = dataset.Outcome
x = dataset.drop(‘Outcome’, axis = 1)
columns = x.columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(x)
data_x = pd.DataFrame(X, columns = columns)

#Splitting the data into training and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x, Y, test_size = 0.15, random_state = 45)

from imblearn.over_sampling import SMOTE
smt = SMOTE()
x_train, y_train = smt.fit_sample(x_train, y_train)
np.bincount(y_train)
Out[74]: array([430, 430])

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
# Accuracy of logistic regression classifier on test set: 0.73

print(f1_score(y_test, y_pred, average=”macro”))
print(precision_score(y_test, y_pred, average=”macro”))
print(recall_score(y_test, y_pred, average=”macro”))
#0.723703419131771
#0.7220530003045994
#0.7263975155279503

#Random Forest 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, bootstrap = True, max_features = ‘sqrt’)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('Accuracy of Random Forest on test set: {:.2f}'.format(model.score(x_test, y_test)))
#Accuracy of Random Forest on test set: 0.88

print(f1_score(y_test, y_pred, average="macro"))
print(precision_score(y_test, y_pred, average="macro"))
print(recall_score(y_test, y_pred, average="macro"))
#0.8729264475743349
#0.8762626262626263
#0.8701863354037267

#We thus select the Random Forest Classifier as the right model due to high accuracy, precision and recall score. 