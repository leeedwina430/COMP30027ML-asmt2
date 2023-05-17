#%%
# load data
import pandas as pd

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

label_var = 'rating_label'

# Gets the data types, and removes all int data types
data[label_var], ind2label = pd.factorize(data[label_var])
data = data.select_dtypes(include="number")

# normalize numerical variables
for feature in data:
    if feature == label_var: continue
    scaler = StandardScaler()
    data[feature] = scaler.fit_transform(data[feature].values.reshape(-1,1))

# NOTE: first try to ignore the categorical variables
# data = pd.get_dummies(data, columns=cat_vars, drop_first=False, prefix=cat_vars)

# concatenate all features
data = pd.concat([data, fea_name, fea_authors, fea_desc], axis=1)
print('total data shape: ',data.shape)
# print(data[label_var].value_counts())



##############
#     PCA    #
##############

#%%
# Method 4: use PCA
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset

# Apply PCA for dimensionality reduction
pca = PCA(n_components=100)  # Select the number of components you want to keep
X_reduced = pca.fit_transform(data.drop(columns=label_var).values)

print(X_reduced.shape)

labels = data[label_var].values
data = pd.DataFrame(data=X_reduced, columns=['pca'+str(i) for i in range(X_reduced.shape[1])])
data[label_var] = labels.astype(int)



#########################################
#     Oversampling and Undersampling    #
#########################################

#%%
# split data into training and validation sets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import nn as my_nn

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

X = data.drop(columns=label_var).values
Y = data[label_var].values
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=0, stratify=Y_temp)

print('trianing set: ', X_train.shape, Y_train.shape)
print('validation set: ', X_val.shape, Y_val.shape)
print('test set: ', X_test.shape, Y_test.shape)

print("\ntraining set class dis\n", pd.Series(Y_train).value_counts().values / len(Y_train))
print("validation set class dis\n", pd.Series(Y_val).value_counts().values / len(Y_val))
print("test set class dis\n", pd.Series(Y_test).value_counts().values / len(Y_test))


#%%
# Method 0: using the equal class weight
loss_fn = nn.CrossEntropyLoss()

trainData = my_nn.MyDataset(X_train, Y_train)
valData = my_nn.MyDataset(X_val, Y_val)
testData = my_nn.MyDataset(X_test, Y_test)


#%%
# Method 1: using the inverse class weight
class_weight = data[label_var].value_counts().sum() / data[label_var].value_counts().values
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float32).to(device))

trainData = my_nn.MyDataset(X_train, Y_train)
valData = my_nn.MyDataset(X_val, Y_val)
testData = my_nn.MyDataset(X_test, Y_test)


#%%
# Method 2: using SMOTE
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from torch.utils.data import TensorDataset
import numpy as np

smote = SMOTE(random_state=42)
# smote = ADASYN(random_state=42)
# smote = BorderlineSMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, Y_train)
X_train_resampled = torch.from_numpy(X_train_resampled).float()
y_train_resampled = torch.from_numpy(y_train_resampled).long()

print(X_train_resampled.shape)
print(np.unique(y_train_resampled, return_counts=True))


smote = BorderlineSMOTE(random_state=42)
X_val_resampled, y_val_resampled = smote.fit_resample(X_val, Y_val)
X_val_resampled = torch.from_numpy(X_val_resampled).float()
y_val_resampled = torch.from_numpy(y_val_resampled).long()

print(X_val_resampled.shape)
print(np.unique(y_val_resampled, return_counts=True))


loss_fn = nn.CrossEntropyLoss()

trainData = TensorDataset(X_train_resampled, y_train_resampled)
valData = TensorDataset(X_val_resampled, y_val_resampled)
# valData = my_nn.MyDataset(X_val, Y_val)
testData = my_nn.MyDataset(X_test, Y_test)


#%%
# Method 3: undersampling
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, Y_train)
X_train_resampled = torch.from_numpy(X_train_resampled).float()
y_train_resampled = torch.from_numpy(y_train_resampled).long()

print(X_train_resampled.shape)
print(np.unique(y_train_resampled, return_counts=True))

loss_fn = nn.CrossEntropyLoss()

trainData = TensorDataset(X_train_resampled, y_train_resampled)
valData = my_nn.MyDataset(X_val, Y_val)
testData = my_nn.MyDataset(X_test, Y_test)

