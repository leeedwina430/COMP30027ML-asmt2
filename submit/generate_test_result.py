#######################################
#  code for generating the test file  #
#######################################


#%%
# load data
import pandas as pd
import numpy as np
import os
print(os.getcwd())
os.chdir('NN')

data_dir = '../data'
train_path = data_dir+"/book_rating_train.csv"
# fea_name_path = data_dir+"/book_text_features_doc2vec/train_name_doc2vec100.csv"
# fea_authors_path = data_dir+"/book_text_features_doc2vec/train_authors_doc2vec20.csv"
# fea_desc_path = data_dir+"/book_text_features_doc2vec/train_desc_doc2vec100.csv"
train_data = pd.read_csv(train_path, index_col = False, delimiter = ',', header=0)

test_path = data_dir+"/book_rating_test.csv"
fea_name_path = data_dir+"/book_text_features_doc2vec/test_name_doc2vec100.csv"
fea_authors_path = data_dir+"/book_text_features_doc2vec/test_authors_doc2vec20.csv"
fea_desc_path = data_dir+"/book_text_features_doc2vec/test_desc_doc2vec100.csv"
data = pd.read_csv(test_path, index_col = False, delimiter = ',', header=0)

fea_name = pd.read_csv(fea_name_path, index_col = False, delimiter = ',', header=None)
fea_authors = pd.read_csv(fea_authors_path, index_col = False, delimiter = ',', header=None)
fea_desc = pd.read_csv(fea_desc_path, index_col = False, delimiter = ',', header=None)

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

# Gets the data types, and removes all int data types
train_data[label_var], ind2label = pd.factorize(train_data[label_var])  # 10
# data = data.select_dtypes(include="number")
data = data.drop(columns=embed_vars)    # 7

# # normalize numerical variables
# for feature in num_vars:
#     scaler = StandardScaler()
#     scaler.fit(data[feature].values.reshape(-1,1))
#     data[feature] = scaler.transform(data[feature].values.reshape(-1,1))

# concatenate all features
data = pd.concat([data, fea_name, fea_authors, fea_desc], axis=1)
print(data.shape)

# # PCA
# from sklearn.decomposition import PCA
# from torch.utils.data import TensorDataset

# # Apply PCA for dimensionality reduction
# pca = PCA(n_components=100)  # Select the number of components you want to keep
# X_reduced = pca.fit_transform(data.drop(columns=[label_var]+cat_vars).values)

# print(X_reduced.shape)

# labels = data[label_var].values
# temp_data = data[cat_vars]
# data = pd.DataFrame(data=X_reduced, columns=['pca'+str(i) for i in range(X_reduced.shape[1])])
# data = pd.concat([data, temp_data], axis=1)
# data[label_var] = labels.astype(int)


num_input_dim = data.shape[1] - len(cat_vars)
save_input_dim = data.shape[1] - len(cat_vars)

# data = pd.get_dummies(data, columns=cat_vars, drop_first=False, prefix=cat_vars)
data[cat_vars[0]+"_num"], v1_ind2label = pd.factorize(data[cat_vars[0]], use_na_sentinel=False)  # 4247
data[cat_vars[1]+"_num"], v2_ind2label = pd.factorize(data[cat_vars[1]], use_na_sentinel=False)  # 10
data = data.drop(columns=cat_vars)    # 7

cat_input_dim = data.shape[1] - save_input_dim

dictionaries = [len(v1_ind2label)+1, len(v2_ind2label)+1]


# split data into training and validation sets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import NeuralNet as my_nn

# X = data.drop(columns=label_var).values
# Y = data[label_var].values
# X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
# X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=0, stratify=Y_temp)
# trainData = my_nn.MyDataset(X_train, Y_train)
# valData = my_nn.MyDataset(X_val, Y_val)
# testData = my_nn.MyDataset(X_test, Y_test)

X = data.values
Y = np.zeros(data.iloc[:,0].values.shape)
testData = my_nn.MyDataset(X, Y)

#%%
# define the model
from torch.utils.tensorboard import SummaryWriter, writer
from utils import save_checkpoint, load_checkpoint 
import NeuralNet as my_nn
import pickle
import time

# training
num = '6_categories_nonnormal_embedlayer_150'
dir = "model"
wfile = "{}/model_{}".format(dir,num)
# writer = SummaryWriter(f"model_{num}.log")
input_dim, output_dim = data.shape[1]-1, len(ind2label)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

hidden_dim = 70
embed_dims = [100,10] # [10, 3]
# EPOCHS = 150
# LR = 1e-3
# BATCH_SIZE = 2048 # 1024
# STEP_SIZE = 10  # 10
# GAMMA = 0.5
# DROPOUT = 0.0
# WEIGHT_DACAY = 1e-3 # refularization term

torch.manual_seed(0)
# trainSet = DataLoader(dataset=trainData, shuffle=True, drop_last=False, batch_size=BATCH_SIZE)
# valSet = DataLoader(dataset=valData, shuffle=False, drop_last=False, batch_size=len(valData))
testSet = DataLoader(dataset=testData, shuffle=False, drop_last=False, batch_size=len(testData))

# load the model
import pickle
model_dir = wfile+'.mdl'
with open(model_dir, 'rb') as f:
    model = pickle.load(f)


#%%
# test
def test(model, testSet):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in testSet:
            x, y = x.to(device), y.to(device)
            output = model(x)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.argmax(output, dim=1).cpu().numpy())
    model.train()
    return y_true, y_pred

y_true, y_pred = test(model, testSet)
y_pred = [ind2label[idx] for idx in y_pred]

print(y_pred)

#%%
# save into the uploading format
outcome = pd.DataFrame({'id':np.arange(len(y_pred))+1, 
                        # label_var:np.array(y_pred, dtype=np.float64)
                        label_var:y_pred
                        })

outcome.to_csv(f'outcome/{num}_test.csv', index=None)
