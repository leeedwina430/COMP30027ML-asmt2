##########################################
#  3 different architectures for the nn  #
##########################################
#%%
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]	

    def __len__(self):
        return len(self.x)


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


class CategoricalClassifier(nn.Module):
    def __init__(self, num_input_dim, dictionaries, embed_dims, hidden_dim, output_dim, dropoutp=0.1):
        super(CategoricalClassifier, self).__init__()
        self.embed1 = nn.Embedding(dictionaries[0],  embed_dims[0])
        self.embed2 = nn.Embedding(dictionaries[1],  embed_dims[1])
        self.fc1 = nn.Linear(num_input_dim + embed_dims[0] + embed_dims[1], hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropoutp)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        X_num = x[:, :-2]
        X_cat1 = x[:, [-2]].T.long()
        X_cat2 = x[:, [-1]].T.long()

        X_cat1 = self.embed1(X_cat1).squeeze(0)
        X_cat2 = self.embed2(X_cat2).squeeze(0)
        X_cat1 = torch.flatten(X_cat1, start_dim=1)
        X_cat2 = torch.flatten(X_cat2, start_dim=1)
        x = torch.cat((X_num, X_cat1, X_cat2), dim=1)

        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out


class CategoricalClassifier3(nn.Module):
    def __init__(self, num_input_dim, dictionaries, embed_dims, hidden_dim, output_dim, dropoutp=0.1):
        super(CategoricalClassifier3, self).__init__()
        self.embed1 = nn.Embedding(dictionaries[0],  embed_dims[0])
        self.embed2 = nn.Embedding(dictionaries[1],  embed_dims[1])
        self.fc1 = nn.Linear(num_input_dim + embed_dims[0] + embed_dims[1], hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=dropoutp)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.dropout2 = nn.Dropout(p=dropoutp)

    def forward(self, x):
        X_num = x[:, :-2]
        X_cat1 = x[:, [-2]].T.long()
        X_cat2 = x[:, [-1]].T.long()

        X_cat1 = self.embed1(X_cat1).squeeze(0)
        X_cat2 = self.embed2(X_cat2).squeeze(0)
        X_cat1 = torch.flatten(X_cat1, start_dim=1)
        X_cat2 = torch.flatten(X_cat2, start_dim=1)
        x = torch.cat((X_num, X_cat1, X_cat2), dim=1)

        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout1(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.dropout2(out)
        out = self.fc4(out)
        return out