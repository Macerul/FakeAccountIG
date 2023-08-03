import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.tree import export_graphviz


'''
# REAL ORIGINAL DATASET
df_real_account = pd.read_csv('out_final/real_account_IG_dataset_original.csv', index_col='id')
df_real_account = df_real_account*1
df_real_account = df_real_account.astype(int)
# adding the target column
df_real_account['isFake'] = 0
df_real_account.to_csv('real_dataset/Zeros_original.csv')
print(df_real_account)
'''


'''
# REAL DATASET (with is_business and is_recent_user)
df_real_account = pd.read_csv('out_final/real_account_IG_dataset.csv', index_col='id')
df_real_account = df_real_account*1
df_real_account = df_real_account.astype(int)
# adding the target column
df_real_account['isFake'] = 0
df_real_account.to_csv('real_dataset/Zeros_with_2columns.csv')
print(df_real_account)
'''


'''
# FAKE ORIGINAL DATASET MOST FREQ
df_fake_account = pd.read_csv('dataset/fake_account_most_freq_imputer_original.csv', index_col='id')
df_fake_account = df_fake_account*1
df_fake_account = df_fake_account.astype(int)
# adding the target column
df_fake_account['isFake'] = 1
df_fake_account.to_csv('fake_dataset/fake_mostfreq_original.csv', header=None)
print(df_fake_account)

'''

'''
# FAKE DATASET MOST FREQ
df_fake_account = pd.read_csv('dataset/fake_account_most_freq_imputer.csv', index_col='id')
df_fake_account = df_fake_account*1
df_fake_account = df_fake_account.astype(int)
# adding the target column
df_fake_account['isFake'] = 1
df_fake_account.to_csv('fake_dataset/fake_mostfreq_with_2columns.csv', header=None)
print(df_fake_account)
'''


'''
# FAKE ORIGINAL DATASET MICE
df_fake_account = pd.read_csv('dataset/fake_account_MICE_imputer_original.csv', index_col='id')
df_fake_account = df_fake_account*1
df_fake_account = df_fake_account.astype(int)
# adding the target column
df_fake_account['isFake'] = 1
df_fake_account.to_csv('fake_dataset/fake_MICE_original.csv', header=None)
print(df_fake_account.isnull().sum())
'''

'''
# FAKE DATASET MICE
df_fake_account = pd.read_csv('dataset/fake_account_MICE_imputer.csv', index_col='id')
df_fake_account = df_fake_account*1
df_fake_account = df_fake_account.astype(int)
# adding the target column
df_fake_account['isFake'] = 1
df_fake_account.to_csv('fake_dataset/fake_MICE_with_2columns.csv', header=None)
print(df_fake_account.isnull().sum())
'''

'''
# FAKE DATASET WITH 0s
df_fake_account = pd.read_csv('out_final/fake_account_IG_dataset.csv', index_col='id')
df_fake_account = df_fake_account*1
df_fake_account = df_fake_account.fillna(0)
df_fake_account = df_fake_account.astype(int)
# adding the target column
df_fake_account['isFake'] = 1
df_fake_account.to_csv('fake_dataset/fake_0s_with_2columns.csv', header=None)
print(df_fake_account.isnull().sum())
'''

'''
# FAKE DATASET ORIGINAL WITH 0s
df_fake_account = pd.read_csv('out_final/fake_account_IG_dataset.csv', index_col='id')
df_fake_account = df_fake_account.drop(['is_business', 'is_recent_user'], axis=1)
df_fake_account = df_fake_account*1
df_fake_account = df_fake_account.fillna(0)
df_fake_account = df_fake_account.astype(int)
# adding the target column
df_fake_account['isFake'] = 1
df_fake_account.to_csv('fake_dataset/fake_0s_original.csv', header=None)
print(df_fake_account.isnull().sum())
'''


# 1st EXPERIMENT REAL(ORIGINAL) - FAke (ORIGINAL) METHOD: Simple Imputer 'most_frequent'
'''
df_real = pd.read_csv('real_dataset/mostfreq_original.csv', index_col='id')
df_real = df_real*1
df_real = df_real.astype(int)
print(df_real.isnull().sum())
print(df_real)
X = df_real.drop(['isFake'], axis=1)
print(X)
y = df_real['isFake']
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''

# 2nd EXPERIMENT REAL(ORIGINAL) - FAke (ORIGINAL) METHOD: Iterative Imputer 'MICE'
'''
df_real = pd.read_csv('real_dataset/MICE_original.csv', index_col='id')
df_real = df_real*1
df_real = df_real.astype(int)
print(df_real.isnull().sum())
print(df_real)
X = df_real.drop(['isFake'], axis=1)
print(X)
y = df_real['isFake']
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''

'''
# 3d EXPERIMENT REAL - FAke  METHOD: Simple Imputer method: 'most_frequent'
df_real = pd.read_csv('real_dataset/most_freq_with_2columns.csv', index_col='id')
df_real = df_real*1
df_real = df_real.astype(int)
print(df_real.isnull().sum())
print(df_real)
X = df_real.drop(['isFake'], axis=1)
print(X)
y = df_real['isFake']
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''

'''
# 4th EXPERIMENT REAL - FAke  METHOD: Iterative Imputer method: 'MICE'
df_real = pd.read_csv('real_dataset/MICE_with_2columns.csv', index_col='id')
df_real = df_real*1
df_real = df_real.astype(int)
print(df_real.isnull().sum())
print(df_real)
X = df_real.drop(['isFake'], axis=1)
print(X)
y = df_real['isFake']
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''

'''
# 5th EXPERIMENT REAL - FAke  USING SMOTE TECNIQUE TO BALANCE BOTH CATEGORIES AND EQUAL AMOUNT OF DATA
# USING MICE DATASET WITH is_business, is_private
df_real = pd.read_csv('real_dataset/MICE_with_2columns.csv', index_col='id')
df_real = df_real*1
df_real = df_real.astype(int)
print(df_real.isnull().sum())
print(df_real)
X = df_real.drop(['isFake'], axis=1)
# print(X)
y = df_real['isFake']
# print(y)

# APPLYING SMOTE TECNIQUE
print(df_real['isFake'].value_counts())
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)


# print(X)
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''

'''
# 6th EXPERIMENT REAL - FAke (Extended dataset) (using fillna(0) strategy)
# instead of using a mostfreq-mice tecnique to compensate for the null values)
df_real = pd.read_csv('real_dataset/Zeros_with_2columns.csv', index_col='id')
df_real = df_real*1
df_real = df_real.astype(int)
print(df_real.isnull().sum())
print(df_real)
X = df_real.drop(['isFake'], axis=1)
print(X)
y = df_real['isFake']
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''


# 7th EXPERIMENT REAL - FAke (ORIGINAL)  (using fillna(0) strategy)
df_real = pd.read_csv('real_dataset/Zeros_original.csv', index_col='id')
df_real = df_real*1
df_real = df_real.astype(int)
print(df_real.isnull().sum())
print(df_real)
X = df_real.drop(['isFake'], axis=1)
print(X)
y = df_real['isFake']
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# KNN
'''
# create new a knn model
knn2 = KNeighborsClassifier()
# create a dictionary of all values we want to test for n_neighbors
param_grid = {
    'n_neighbors': np.arange(1, 25),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}
# use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
# fit model to data
knn_gscv.fit(X_train, y_train)

print('best score:',  knn_gscv.best_score_, 'best params', knn_gscv.best_params_)

'''

'''
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=6, p=1, weights='distance')
# Fit the classifier to the data
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy for kNN on Test data: ", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred, labels=knn.classes_)
print(conf_matrix)
df_cm = pd.DataFrame(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=knn.classes_)
plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=True, fmt=".0f")
plt.title('Confusion Matrix Correlation-Coefficient')
disp.plot()
plt.show()
'''

# RANDOM FOREST
'''
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000],
    'criterion': ['gini', 'entropy']
}


# Create a based model
rfc = RandomForestClassifier(random_state=42)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

print('Best param: ', grid_search.best_params_, 'Best score: ', grid_search.best_score_)
'''

'''
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42,
                                                                  bootstrap=True,
                                                                  criterion='entropy',
                                                                  max_depth=80,
                                                                  max_features=3,
                                                                  min_samples_leaf=3,
                                                                  min_samples_split=8,
                                                                  n_estimators=1000))

# Pass instance of pipeline and training and test data set
scores = cross_val_score(pipeline, X=X_train, y=y_train, cv=10, n_jobs=1)
print('Cross Validation accuracy scores: %s' % scores)
'''

'''
# new train with best hyperparameters
rfc1 = RandomForestClassifier(random_state=42,
                              bootstrap=True,
                              criterion='entropy',
                              max_depth=80,
                              max_features=2,
                              min_samples_leaf=3,
                              min_samples_split=8,
                              n_estimators=100)

rfc1.fit(X_train, y_train)
y_pred = rfc1.predict(X_test)

# view the feature scores
feature_scores = pd.Series(rfc1.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print('Feature score: ', feature_scores)

print("Accuracy for Random Forest on Test data: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred, labels=rfc1.classes_)
print(conf_matrix)
df_cm = pd.DataFrame(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=rfc1.classes_)
plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=True, fmt=".0f")
plt.title('Confusion Matrix Correlation-Coefficient')
disp.plot()
plt.show()
'''


# DECISION TREE

'''
params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=5)
grid_search.fit(X_train, y_train)
print('Best param: ', grid_search.best_params_, 'Best score: ', grid_search.best_score_)
'''


'''
# new train with best hyperparameters
dt = DecisionTreeClassifier(random_state=42, max_leaf_nodes=15, min_samples_split=2)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print("Accuracy for Decision Tree on Test data: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred, labels=dt.classes_)
print(conf_matrix)
df_cm = pd.DataFrame(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=dt.classes_)
plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=True, fmt=".0f")
plt.title('Confusion Matrix Correlation-Coefficient')
disp.plot()
plt.show()
'''

# GAUSSIAN NAIVE BAYES
'''
nb_classifier = GaussianNB()

params_NB = {'var_smoothing': np.logspace(0, -9, num=100)}

gs_NB = GridSearchCV(estimator=nb_classifier,
                     param_grid=params_NB,
                     cv=5,   # use any cross validation technique
                     verbose=1,
                     scoring='accuracy')
gs_NB.fit(X_train, y_train)

print('Best params: ', gs_NB.best_params_, 'Best score: ', gs_NB.best_score_)
'''

'''
# new train with best hyperparameters
nb = GaussianNB(var_smoothing=1e-09)
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

print("Accuracy for Gaussian Naive Bayes on Test data: ", accuracy_score(y_test,y_pred))
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred, labels=nb.classes_)
print(conf_matrix)
df_cm = pd.DataFrame(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=nb.classes_)
plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=True, fmt=".0f")
plt.title('Confusion Matrix Correlation-Coefficient')
disp.plot()
plt.show()
'''

# SVM
'''
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, cv=5)

# fitting the model for grid search
grid.fit(X_train, y_train)

print('Best params: ', grid.best_params_, 'Best score: ', grid.best_score_)
'''

'''
# new train with best hyperparameters
svc = SVC(C=1, gamma=0.0001, kernel='rbf')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

print("Accuracy for SVC on Test data: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred, labels=svc.classes_)
print(conf_matrix)
df_cm = pd.DataFrame(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=svc.classes_)
plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=True, fmt=".0f")
plt.title('Confusion Matrix Correlation-Coefficient')
disp.plot()
plt.show()
'''

# LOGISTIC REGRESSION
'''
param_grid = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

lf = LogisticRegression(solver='liblinear', max_iter=10000)

grid = GridSearchCV(estimator=lf, param_grid=param_grid, scoring='accuracy', verbose=3, n_jobs=-1, cv=5)

grid.fit(X_train, y_train)
print('Best params: ', grid.best_params_, 'Best score: ', grid.best_score_)
'''
'''
# new train with best hyperparameters
lr = LogisticRegression(solver='liblinear', C=0.001, penalty='l2')
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("Accuracy for Logistic Regression on Test data: ", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred, labels=lr.classes_)
print(conf_matrix)
df_cm = pd.DataFrame(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=lr.classes_)
plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=True, fmt=".0f")
plt.title('Confusion Matrix Correlation-Coefficient')
disp.plot()
plt.show()
'''










