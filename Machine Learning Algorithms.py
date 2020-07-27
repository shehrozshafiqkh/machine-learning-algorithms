#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data analysis
import pandas as pd
import numpy as np
import pickle

# data visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning models
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# data splitting
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# evaluation measures
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# ### Data Preprocessing

# In[159]:


# reading titanic-dataset.csv file data
df = pd.read_csv('Dataset\titanic-dataset.csv')
df.head()


# In[160]:


# extracting features and target of our use
df = df[['Sex', 'Pclass', 'SibSp', 'Embarked', 'Survived']]
df.head()


# In[161]:


# function to convert categorical values into integers
def convert_sex_to_int(word):
    word_dict = {
        'female': 0,
        'male': 1
    }
    return word_dict[word]


# In[162]:


# converting categorical values into integers
df['Sex'] = df['Sex'].apply(lambda x: convert_sex_to_int(x))
df.head()


# In[163]:


# function to convert categorical values into integers
def convert_embarked_to_int(word):
    word_dict = {
        'S': 0,
        'C': 1,
        'Q': 2
    }
    return word_dict[word]


# In[164]:


# checking null values
df.isnull().sum()


# In[166]:


# dataframe information
df.info()


# In[171]:


# dropping null values
df.dropna(axis=0, inplace=True)


# In[172]:


# checking null values
df.isnull().sum()


# In[173]:


# dataframe information
df.info()


# In[174]:


# converting categorical values into integers
df['Embarked'] = df['Embarked'].apply(lambda x: convert_embarked_to_int(x))
df.head()


# In[175]:


# column names of training data
df.columns.values


# In[176]:


# dataframe/dataset information
df.info()


# In[177]:


# dataset describtion
df.describe()


# In[183]:


# checking the shape of dataset/dataframe
df.shape


# In[561]:


# correlation plot
sns.heatmap(df.corr())
plt.show()


# ### Train Test Split

# In[191]:


# featurization
train_df = df.iloc[:, 0:4]
train_df.head()


# In[192]:


# featurization
test_df = df['Survived']
test_df.head()


# In[199]:


# train test split
X_train, X_test, y_train, y_test = train_test_split(train_df, test_df, test_size=0.2, random_state=101)


# In[200]:


X_train.shape


# In[196]:


y_train.shape


# In[201]:


X_test.shape


# In[197]:


y_test.shape


# ### Machine Learning Algorithms

# In[319]:


# Linear Regression
linear_regression = LinearRegression()

# training the model
linear_regression.fit(X_train, y_train)

# making predictions on test data
predictions = linear_regression.predict(X_test)

# evaluation measures
acc_linear_regression = accuracy_score(y_test, predictions.round())
pre_linear_regression = precision_score(y_test, predictions.round())
rec_linear_regression = recall_score(y_test, predictions.round())
f1_linear_regression = f1_score(y_test, predictions.round())


# In[297]:


# Logistic Regression
logistic_regression = LogisticRegression()

# training the model
logistic_regression.fit(X_train, y_train)

# making predictions on test data
predictions = logistic_regression.predict(X_test)

# evaluation measures
acc_logistic_regression = accuracy_score(y_test, predictions)
pre_logistic_regression = precision_score(y_test, predictions)
rec_logistic_regression = recall_score(y_test, predictions)
f1_logistic_regression = f1_score(y_test, predictions)


# In[296]:


# Random Forest
random_forest = RandomForestClassifier()

# training the model
random_forest.fit(X_train, y_train)

# making predictions on test data
predictions = random_forest.predict(X_test)

# evaluation measures
acc_random_forest = accuracy_score(y_test, predictions)
pre_random_forest = precision_score(y_test, predictions)
rec_random_forest = recall_score(y_test, predictions)
f1_random_forest = f1_score(y_test, predictions)


# In[344]:


# K Neighbors
knn = KNeighborsClassifier()

# training the model
knn.fit(X_train, y_train)

# making predictions on test data
predictions = knn.predict(X_test)

# evaluation measures
acc_knn = accuracy_score(y_test, predictions)
pre_knn = precision_score(y_test, predictions)
rec_knn = recall_score(y_test, predictions)
f1_knn = f1_score(y_test, predictions)


# In[294]:


# Gaussian Naive Bayes
naive_bayes = GaussianNB()

# training the model
naive_bayes.fit(X_train, y_train)

# making predictions on test data
predictions = naive_bayes.predict(X_test)

# evaluation measures
acc_naive_bayes = accuracy_score(y_test, predictions)
pre_naive_bayes = precision_score(y_test, predictions)
rec_naive_bayes = recall_score(y_test, predictions)
f1_naive_bayes = f1_score(y_test, predictions)


# In[293]:


# Decision Tree
decision_tree = DecisionTreeClassifier()

# training the model
decision_tree.fit(X_train, y_train)

# making predictions on test data
predictions = decision_tree.predict(X_test)

# evaluation measures
acc_decision_tree = accuracy_score(y_test, predictions)
pre_decision_tree = precision_score(y_test, predictions)
rec_decision_tree = recall_score(y_test, predictions)
f1_decision_tree = f1_score(y_test, predictions)


# In[407]:


# Support Vector Machine
svc = SVC()

# training the model
svc.fit(X_train, y_train)

# making predictions on test data
predictions = svc.predict(X_test)

# evaluation measures
acc_svc = accuracy_score(y_test, predictions)
pre_svc = precision_score(y_test, predictions)
rec_svc = recall_score(y_test, predictions)
f1_svc = f1_score(y_test, predictions)


# In[291]:


# Linear SVC
linear_svc = LinearSVC()

# training the model
linear_svc.fit(X_train, y_train)

# making predictions on test data
predictions = linear_svc.predict(X_test)

# evaluation measures
acc_linear_svc = accuracy_score(y_test, predictions)
pre_linear_svc = precision_score(y_test, predictions)
rec_linear_svc = recall_score(y_test, predictions)
f1_linear_svc = f1_score(y_test, predictions)


# In[290]:


# Stochastic Gradient Descent
sgd = SGDClassifier()

# training the model
sgd.fit(X_train, y_train)

# making predictions on test data
predictions = sgd.predict(X_test)

# evaluation measures
acc_sgd = accuracy_score(y_test, predictions)
pre_sgd = precision_score(y_test, predictions)
rec_sgd = recall_score(y_test, predictions)
f1_sgd = f1_score(y_test, predictions)


# In[386]:


# Perceptron
perceptron = Perceptron()

# training the model
perceptron.fit(X_train, y_train)

# making predictions on test data
predictions = perceptron.predict(X_test)

# evaluation measures
acc_perceptron = accuracy_score(y_test, predictions)
pre_perceptron = precision_score(y_test, predictions)
rec_perceptron = recall_score(y_test, predictions)
f1_perceptron = f1_score(y_test, predictions)


# In[533]:


# making dataframe of evaluation measures
models_traintestsplit = pd.DataFrame({
    'Model': ['Linear Regression', 'Logistic Regression', 'Random Forest', 'K Neighbors', 'Naive Bayes', 'Decision Tree', 'Support Vector Machine', 'Linear SVC', 'Stochastic Gradient Descent', 'Perceptron'],
    'Accuracy': [acc_linear_regression, acc_logistic_regression, acc_random_forest, acc_knn, acc_naive_bayes, acc_decision_tree, acc_svc, acc_linear_svc, acc_sgd, acc_perceptron],
    'Precision': [pre_linear_regression, pre_logistic_regression, pre_random_forest, pre_knn, pre_naive_bayes, pre_decision_tree, pre_svc, pre_linear_svc, pre_sgd, pre_perceptron],
    'Recall': [rec_linear_regression, rec_logistic_regression, rec_random_forest, rec_knn, rec_naive_bayes, rec_decision_tree, rec_svc, rec_linear_svc, rec_sgd, rec_perceptron],
    'F1': [f1_linear_regression, f1_logistic_regression, f1_random_forest, f1_knn, f1_naive_bayes, f1_decision_tree, f1_svc, f1_linear_svc, f1_sgd, f1_perceptron]
})

# sorting values
models_traintestsplit.sort_values(['Accuracy', 'Precision', 'Recall', 'F1'], ascending=False)


# ### K-Fold Cross Validation

# In[516]:


# Linear Regression
linear_regression_kfold = LinearRegression()
train_df = np.array(train_df)
test_df = np.array(test_df)
# perform 10-fold Cross Validation
acc_linear_regression_kfold = np.sqrt(-cross_val_score(linear_regression_kfold, train_df, test_df, cv=10, scoring='neg_mean_squared_error')).mean()


# ******************************************** Note ********************************************
# ****** Linear Regression is a regression problem and cant be evaluated using accuracy. *******
# ****************** We need root meat squared error to evaluate predictions. ******************


# In[517]:


# Logistic Regression
logistic_regression_kfold = LogisticRegression()

# perform 10-fold Cross Validation
acc_logistic_regression_kfold = cross_val_score(logistic_regression_kfold, train_df, test_df, cv=10, scoring='accuracy').mean()
pre_logistic_regression_kfold = cross_val_score(logistic_regression_kfold, train_df, test_df, cv=10, scoring='precision').mean()
rec_logistic_regression_kfold = cross_val_score(logistic_regression_kfold, train_df, test_df, cv=10, scoring='recall').mean()
f1_logistic_regression_kfold = cross_val_score(logistic_regression_kfold, train_df, test_df, cv=10, scoring='f1').mean()


# In[518]:


# Random Forest
random_forest_kfold = RandomForestClassifier()

# perform 10-fold Cross Validation
acc_random_forest_kfold = cross_val_score(random_forest_kfold, train_df, test_df, cv=10, scoring='accuracy').mean()
pre_random_forest_kfold = cross_val_score(random_forest_kfold, train_df, test_df, cv=10, scoring='precision').mean()
rec_random_forest_kfold = cross_val_score(random_forest_kfold, train_df, test_df, cv=10, scoring='recall').mean()
f1_random_forest_kfold = cross_val_score(random_forest_kfold, train_df, test_df, cv=10, scoring='f1').mean()


# In[519]:


# K Neighbors
knn_kfold = KNeighborsClassifier()

# perform 10-fold Cross Validation
acc_knn_kfold = cross_val_score(knn_kfold, train_df, test_df, cv=10, scoring='accuracy').mean()
pre_knn_kfold = cross_val_score(knn_kfold, train_df, test_df, cv=10, scoring='precision').mean()
rec_knn_kfold = cross_val_score(knn_kfold, train_df, test_df, cv=10, scoring='recall').mean()
f1_knn_kfold = cross_val_score(knn_kfold, train_df, test_df, cv=10, scoring='f1').mean()


# In[520]:


# Naive Bayes
naive_bayes_kfold = GaussianNB()

# perform 10-fold Cross Validation
acc_naive_bayes_kfold = cross_val_score(naive_bayes_kfold, train_df, test_df, cv=10, scoring='accuracy').mean()
pre_naive_bayes_kfold = cross_val_score(naive_bayes_kfold, train_df, test_df, cv=10, scoring='precision').mean()
rec_naive_bayes_kfold = cross_val_score(naive_bayes_kfold, train_df, test_df, cv=10, scoring='recall').mean()
f1_naive_bayes_kfold = cross_val_score(naive_bayes_kfold, train_df, test_df, cv=10, scoring='f1').mean()


# In[521]:


# Decision Tree
decision_tree_kfold = DecisionTreeClassifier()

# perform 10-fold Cross Validation
acc_decision_tree_kfold = cross_val_score(decision_tree_kfold, train_df, test_df, cv=10, scoring='accuracy').mean()
pre_decision_tree_kfold = cross_val_score(decision_tree_kfold, train_df, test_df, cv=10, scoring='precision').mean()
rec_decision_tree_kfold = cross_val_score(decision_tree_kfold, train_df, test_df, cv=10, scoring='recall').mean()
f1_decision_tree_kfold = cross_val_score(decision_tree_kfold, train_df, test_df, cv=10, scoring='f1').mean()


# In[522]:


# Support Vector Machine
svc_kfold = SVC()

# perform 10-fold Cross Validation
acc_svc_kfold = cross_val_score(svc_kfold, train_df, test_df, cv=10, scoring='accuracy').mean()
pre_svc_kfold = cross_val_score(svc_kfold, train_df, test_df, cv=10, scoring='precision').mean()
rec_svc_kfold = cross_val_score(svc_kfold, train_df, test_df, cv=10, scoring='recall').mean()
f1_svc_kfold = cross_val_score(svc_kfold, train_df, test_df, cv=10, scoring='f1').mean()


# In[523]:


# Linear SVC
linear_svc_kfold = LinearSVC()

# perform 10-fold Cross Validation
acc_linear_svc_kfold = cross_val_score(linear_svc_kfold, train_df, test_df, cv=10, scoring='accuracy').mean()
pre_linear_svc_kfold = cross_val_score(linear_svc_kfold, train_df, test_df, cv=10, scoring='precision').mean()
rec_linear_svc_kfold = cross_val_score(linear_svc_kfold, train_df, test_df, cv=10, scoring='recall').mean()
f1_linear_svc_kfold = cross_val_score(linear_svc_kfold, train_df, test_df, cv=10, scoring='f1').mean()


# In[524]:


# Stochastic Gradient Descent
sgd_kfold = SGDClassifier()

# perform 10-fold Cross Validation
acc_sgd_kfold = cross_val_score(sgd_kfold, train_df, test_df, cv=10, scoring='accuracy').mean()
pre_sgd_kfold = cross_val_score(sgd_kfold, train_df, test_df, cv=10, scoring='precision').mean()
rec_sgd_kfold = cross_val_score(sgd_kfold, train_df, test_df, cv=10, scoring='recall').mean()
f1_sgd_kfold = cross_val_score(sgd_kfold, train_df, test_df, cv=10, scoring='f1').mean()


# In[525]:


# Perceptron
perceptron_kfold = Perceptron()

# perform 10-fold Cross Validation
acc_perceptron_kfold = cross_val_score(perceptron_kfold, train_df, test_df, cv=10, scoring='accuracy').mean()
pre_perceptron_kfold = cross_val_score(perceptron_kfold, train_df, test_df, cv=10, scoring='precision').mean()
rec_perceptron_kfold = cross_val_score(perceptron_kfold, train_df, test_df, cv=10, scoring='recall').mean()
f1_perceptron_kfold = cross_val_score(perceptron_kfold, train_df, test_df, cv=10, scoring='f1').mean()


# In[532]:


# making dataframe of evaluation measures
models_kfold = pd.DataFrame({
    'Model': ['Linear Regression', 'Logistic Regression', 'Random Forest', 'K Neighbors', 'Naive Bayes', 'Decision Tree', 'Support Vector Machine', 'Linear SVC', 'Stochastic Gradient Descent', 'Perceptron'],
    'Accuracy': [acc_linear_regression_kfold, acc_logistic_regression_kfold, acc_random_forest_kfold, acc_knn_kfold, acc_naive_bayes_kfold, acc_decision_tree_kfold, acc_svc_kfold, acc_linear_svc_kfold, acc_sgd_kfold, acc_perceptron_kfold],
    'Precision': ['', pre_logistic_regression_kfold, pre_random_forest_kfold, pre_knn_kfold, pre_naive_bayes_kfold, pre_decision_tree_kfold, pre_svc_kfold, pre_linear_svc_kfold, pre_sgd_kfold, pre_perceptron_kfold],
    'Recall': ['', rec_logistic_regression_kfold, rec_random_forest_kfold, rec_knn_kfold, rec_naive_bayes_kfold, rec_decision_tree_kfold, rec_svc_kfold, rec_linear_svc_kfold, rec_sgd_kfold, rec_perceptron_kfold],
    'F1': ['', f1_logistic_regression_kfold, f1_random_forest_kfold, f1_knn_kfold, f1_naive_bayes_kfold, f1_decision_tree_kfold, f1_svc_kfold, f1_linear_svc_kfold, f1_sgd_kfold, f1_perceptron_kfold]
})

# sorting values
models_kfold.sort_values(['Accuracy', 'Precision', 'Recall', 'F1'], ascending=False)


# In[537]:


# to display both dataframes in same row
from IPython.display import display, HTML

CSS = """
.output {
    flex-direction: row;
}
"""

HTML('<style>{}</style>'.format(CSS))


# In[539]:


# displaying both dataframes
display(models_traintestsplit.sort_values(['Accuracy', 'Precision', 'Recall', 'F1'], ascending=False))
display(models_kfold.sort_values(['Accuracy', 'Precision', 'Recall', 'F1'], ascending=False))


# In[550]:


# save predictions in a csv file
models_traintestsplit.sort_values(['Accuracy', 'Precision', 'Recall', 'F1'], ascending=False).to_csv('Predictions Train Test Split.csv', index=False)
models_kfold.sort_values(['Accuracy', 'Precision', 'Recall', 'F1'], ascending=False).to_csv('Predictions Kfold.csv', index=False)


# In[3]:


# save the model to disk
filename = 'model.pkl'
pickle.dump(random_forest, open(filename, 'wb'))


# In[5]:


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print('Accuracy: %.2f' %(result * 100) + '%')

