#Creat by Jackson 01/07/2020


import pandas as pd
import numpy as np
#Splitthe datasetinto train and test data
from sklearn.model_selection import train_test_split
#Building and training the model
from sklearn.neighbors import KNeighborsClassifier
#k-Fold Cross-Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#read in the data using pandas
df = pd.read_csv('data/diabetes_data.csv')
X = df.drop(columns=['diabetes'])
y = df['diabetes'].values
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)
# Fit the classifier to the data
knn.fit(X_train,y_train)

#Testing the model
#show first 5 model predictions on the test data
knn.predict(X_test)[0:5]

#check accuracy of our model on the test data
knn.score(X_test, y_test)


#k-Fold Cross-Validation
#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=3)
#train model with cv of 5
cv_scores = cross_val_score(knn_cv, X, y, cv=5)

#print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))

#create new a knn model
knn2 = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X, y)
#check top performing n_neighbors value
knn_gscv.best_params_
#check mean score for the top performing value of n_neighbors
knn_gscv.best_score_
print(knn_gscv.best_score_)
print(knn_gscv.best_params_)

# it's can be found that 14 is the optimal value for ‘n_neighbors’.