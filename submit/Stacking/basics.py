#################################################################
#  code for trying on all the classifiers with default setting  #
#################################################################

#%%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#%%
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

    return accuracy_score(y_true, y_pred)

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

# TODO: search for the best way to deal with datetime variable
# all_vars = ['Name', 'Authors', 'PublishYear', 'PublishMonth', 'PublishDay',
#        'Publisher', 'Language', 'pagesNumber', 'Description', 'rating_label']
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
print(data.shape)


#%%
# split data into training and validation sets
from sklearn.model_selection import train_test_split

X = data.drop(columns=label_var).values
Y = data[label_var].values
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=0, stratify=Y_temp)

#%%
# try for other classifiers

results = {}

#%%
# dummy classifier
from sklearn.dummy import DummyClassifier
from collections import Counter

zero_r = DummyClassifier(strategy="most_frequent")
zero_r.fit(X_train, Y_train)
zr_pred = zero_r.predict(X_val)


# confirm it's a 0_R classifier
label_counter = Counter(zr_pred)
print(label_counter.most_common())

acc = showMetrics(Y_val, zr_pred)
results[str(zero_r)] = [acc, zr_pred]

# np.savetxt(f"{zero_r}_pred.csv", zr_pred, delimiter=",")


#%%
# weighted random classifier
stratified_zero_r = DummyClassifier(strategy='stratified')
stratified_zero_r.fit(X_train, Y_train)
# stratified_zr_pred = stratified_zero_r.predict(X_val)

# showMetrics(Y_val, stratified_zr_pred)

accuracies = []
num_runs = 100
for i in range(num_runs):
    acc = stratified_zero_r.score(X_val, Y_val)
    accuracies.append(acc)
# print(accuracies)
print('Average accuracy over {} runs is: {}.'.format(num_runs, np.mean(accuracies)))
results[str(stratified_zero_r)] = [np.mean(accuracies), None]

#%%
from sklearn.naive_bayes import GaussianNB

# Initialize the Naive Bayes classifier
nb_clf = GaussianNB()
nb_clf.fit(X_train, Y_train)
nb_pred = nb_clf.predict(X_val)

acc = showMetrics(Y_val, nb_pred)
results[str(zero_r)] = [acc, zr_pred]


#%%
# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Create the model
LR_clf = LogisticRegression(max_iter=1000)

# Fit the model to the data
LR_clf.fit(X_train, Y_train)
LR_pred = LR_clf.predict(X_val)

acc = showMetrics(Y_val, LR_pred)
results[str(LR_clf)] = [acc, LR_pred]

#%%
# One R
from sklearn.tree import DecisionTreeClassifier
one_r = DecisionTreeClassifier(max_depth=1)
one_r.fit(X_train, Y_train)
one_r_pred = one_r.predict(X_val)

showMetrics(Y_val, one_r_pred)
results[str(one_r)] = [acc, one_r_pred]

#%%
# show the importance
importances = one_r.feature_importances_
print("importances for 1R: ", importances, '\n')
max_index = np.argmax(importances)
best_feature_name = data.columns[max_index]
print(best_feature_name)

# show the results
ybar = one_r.predict(X)
best_feature = X[:, max_index]
plt.scatter(best_feature, ybar, c=ybar)
plt.xlabel(best_feature_name)
plt.ylabel('predicted class')
plt.show()
#print(ybar)

#%%
# Decision Tree
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, Y_train)
dt_pred = dt_clf.predict(X_val)

acc = showMetrics(Y_val, dt_pred)
results[str(dt_clf)] = [acc, dt_pred]


#%%
# Random Forest
from sklearn.ensemble import RandomForestClassifier

RF_clf = RandomForestClassifier()
RF_clf.fit(X_train, Y_train)
RF_pred = RF_clf.predict(X_val)

acc = showMetrics(Y_val, RF_pred)
results[str(RF_clf)] = [acc, RF_pred]

#%%
# Knn
from sklearn.neighbors import KNeighborsClassifier
knn5_clf = KNeighborsClassifier(n_neighbors=5)
knn5_clf.fit(X_train, Y_train)
knn5_pred = knn5_clf.predict(X_val)

acc = showMetrics(Y_val, knn5_pred)
results[str(knn5_clf)] = [acc, knn5_pred]

#%%
knn_clf = KNeighborsClassifier(n_neighbors=33)  # highest
knn_clf.fit(X_train, Y_train)
knn_pred = knn_clf.predict(X_val)

acc = showMetrics(Y_val, knn_pred)
results[str(knn_clf)] = [acc, knn_pred]

#%%
from sklearn.neural_network import MLPClassifier

# Create the model
MLP_clf = MLPClassifier()
MLP_clf.fit(X_train, Y_train)
MLP_pred = MLP_clf.predict(X_val)

acc = showMetrics(Y_val, MLP_pred)
results[str(MLP_clf)] = [acc, MLP_pred]

#%%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Create the model
LDA_clf = LinearDiscriminantAnalysis()
LDA_clf.fit(X_train, Y_train)
LDA_pred = LDA_clf.predict(X_val)

acc = showMetrics(Y_val, LDA_pred)
results[str(LDA_clf)] = [acc, LDA_pred]

#%%
from sklearn.ensemble import AdaBoostClassifier

# Create the model
AdaBoost_clf = AdaBoostClassifier()
AdaBoost_clf.fit(X_train, Y_train)
AdaBoost_pred = AdaBoost_clf.predict(X_val)

acc = showMetrics(Y_val, AdaBoost_pred)
results[str(AdaBoost_clf)] = [acc, AdaBoost_pred]

#%%
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Create the model
gBoost_clf = GradientBoostingClassifier()
gBoost_clf.fit(X_train, Y_train)
gBoost_pred = gBoost_clf.predict(X_val)

acc = showMetrics(Y_val, gBoost_pred)
results[str(gBoost_clf)] = [acc, gBoost_pred]


#%%
# Combination 1: Voting meta
from sklearn.ensemble import VotingClassifier
clf1 = LogisticRegression(multi_class='multinomial', random_state=1, max_iter=1000)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

eclf1 = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
eclf1 = eclf1.fit(X_train, Y_train)
eclf1_pred = eclf1.predict(X_val)

acc = showMetrics(Y_val, eclf1_pred)
results[str(eclf1)] = [acc, eclf1_pred]

#%%

np.array_equal(eclf1.named_estimators_.lr.predict(X_val),
               eclf1.named_estimators_['lr'].predict(X_val))


#%%
eclf2 = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
        voting='soft')
eclf2 = eclf2.fit(X_train, Y_train)
eclf2_pred = eclf2.predict(X_val)

acc = showMetrics(Y_val, eclf2_pred)
results[str(eclf2)] = [acc, eclf2_pred]


#%%
for results_name in results.keys():
    print(f"{results_name} : \nacc={results[results_name][0]}")

