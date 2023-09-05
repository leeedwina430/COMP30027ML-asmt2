#%%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# define the plotting function
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def showMetrics(y_true, y_pred):
    # print the metrics
    print("Accuracy: {:.3f}".format(accuracy_score(y_true, y_pred)))
    print("Precision: {:.3f}".format(precision_score(y_true, y_pred, average='macro', zero_division=0)))
    print("Recall: {:.3f}".format(recall_score(y_true, y_pred, average='macro')))
    print("F1: {:.3f}".format(f1_score(y_true, y_pred, average='macro')))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.show()


#%%
# load data
import pandas as pd
import numpy as np

data_dir = '../data'
train_path = data_dir+"/book_rating_train.csv"
fea_name_path = data_dir+"/book_text_features_doc2vec/train_name_doc2vec100.csv"
fea_authors_path = data_dir+"/book_text_features_doc2vec/train_authors_doc2vec20.csv"
fea_desc_path = data_dir+"/book_text_features_doc2vec/train_desc_doc2vec100.csv"
# test_path = "data/book_rating_test.csv"
data = pd.read_csv(train_path, index_col = False, delimiter = ',', header=0)
fea_name = pd.read_csv(fea_name_path, index_col = False, delimiter = ',', header=None)
fea_authors = pd.read_csv(fea_authors_path, index_col = False, delimiter = ',', header=None)
fea_desc = pd.read_csv(fea_desc_path, index_col = False, delimiter = ',', header=None)
# test_data = pd.read_csv(test_path, index_col = False, delimiter = ',', header=0)


#%%
# transform data into numerical inputs
from sklearn.preprocessing import StandardScaler

num_vars = ['PublishYear', 'PublishMonth', 'PublishDay', 'pagesNumber']
cat_vars = ['Publisher', 'Language']
embed_vars = ['Name', 'Authors', 'Description']
label_var = 'rating_label'

data[label_var], ind2label = pd.factorize(data[label_var])
data = data.select_dtypes(include="number")

# normalize numerical variables
for feature in data:
    if feature == label_var: continue
    scaler = StandardScaler()
    data[feature] = scaler.fit_transform(data[feature].values.reshape(-1,1))

# NOTE: first try to ignore the categorical variables
# data = pd.get_dummies(data, columns=cat_vars, drop_first=False, prefix=cat_vars)
data = pd.concat([data, fea_name, fea_authors, fea_desc], axis=1)

# split data into training and validation sets
from sklearn.model_selection import train_test_split

X = data.drop(columns=label_var).values
Y = data[label_var].values
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=0, stratify=Y_temp)

#%%
#########################
# Model 0:              #
# Meta : LR             #
# Base : KNN, GNB, RF   #
#########################

# Initializing models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import GridSearchCV

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
# clf3 = GaussianNB()
clf3 = AdaBoostClassifier()
clf4 = GradientBoostingClassifier()
lr = LogisticRegression()
# sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4], 
#                           meta_classifier=lr)
sclf = StackingClassifier(classifiers=[clf1, clf2], 
                          meta_classifier=lr)

# access all possible hyper-parameters
sclf.get_params().keys()

#%%
import time

start = time.time()
# grid search for the best hyper-parameter combination
params = {'kneighborsclassifier__n_neighbors': [43],
          'randomforestclassifier__n_estimators': [50],
          'meta_classifier__C': [0.1]}


# start = time.time()
# # grid search for the best hyper-parameter combination
# params = {'kneighborsclassifier__n_neighbors': [43],
#           'randomforestclassifier__n_estimators': [20, 30, 50],
#           'adaboostclassifier__n_estimators': [150],
#           'gradientboostingclassifier__n_estimators': [100, 80, 120],
#           'gradientboostingclassifier__max_depth': [1, 2, 3, 4],
#           'meta_classifier__C': [0.1],
#           }


grid = GridSearchCV(estimator=sclf, 
                    param_grid=params, 
                    cv=5,
                    refit=True,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=2,
                    )
grid.fit(X_train, Y_train)

cv_keys = ('mean_test_score', 'std_test_score', 'params')

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_[cv_keys[0]][r],
             grid.cv_results_[cv_keys[1]][r] / 2.0,
             grid.cv_results_[cv_keys[2]][r]))

print('Time elapsed: %.2f s' % (time.time() - start))
print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)

# evaluate the model on validation set
print("Validation Set:")
Y_pred = grid.predict(X_val)
showMetrics(Y_val, Y_pred)

print("Test Set:")
Y_pred = grid.predict(X_test)
showMetrics(Y_test, Y_pred)

# save the model
import pickle
model_dir = 'models'
with open(model_dir+'/stacking_model0_base_whole_naive.pkl', 'wb') as f:
    pickle.dump(grid, f)
