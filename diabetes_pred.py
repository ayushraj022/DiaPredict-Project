import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#Data collection and analysis of PIMA Diabetes Dataset

#loading data set to a pandas dataframe
diabetes_dataset = pd.read_csv('/Users/tanyaanuja/Desktop/Machine Learning/diabetes.csv')

#print first 5 row of dataset
diabetes_dataset.head()

#number of row and column of dataset
diabetes_dataset.shape

#getting the stat measure of data
diabetes_dataset.describe()
	
#diabetes outcome
# 0 = diabetic
# 1 = non diabetic
diabetes_dataset["Outcome"].value_counts()

diabetes_dataset.groupby("Outcome").mean()

# seperating data and labels
X = diabetes_dataset.drop(columns = "Outcome", axis = 1)
Y = diabetes_dataset['Outcome']

print(X)	
print(Y)

#Data Standardization
scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

X = standardized_data
Y = diabetes_dataset["Outcome"]

print(X)
print(Y)

#Train Test Split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, stratify = Y, random_state = 2)

print(X.shape, X_train.shape, X_test.shape)

#Training the Model
classifier = svm.SVC(kernel='linear')

#training the SVM classifier
classifier.fit(X_train, Y_train)

#Module Evaluation

# Accuracy Score on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Accuracy score of training data: ", training_data_accuracy)

# Accuracy Score on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Accuracy score of test data: ", test_data_accuracy)

#Making a predictive system**

input_data = (1,110,92,0,0,37.6,0.191,30)

#changing data into numpy array

input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if prediction[0] == 0:
  print("The person is not diabetic")
else:
  print("The person is diabetic")
