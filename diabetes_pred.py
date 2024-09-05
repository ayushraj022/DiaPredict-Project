
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score



#Data collection and analysis

#PIMA Diabetes Dataset

#loading data set to a pandas dataframe
diabetes_dataset = pd.read_csv('/content/diabetes.csv')


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
