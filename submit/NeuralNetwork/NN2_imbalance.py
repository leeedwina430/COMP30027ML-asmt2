############################################
#  try different sampling methods with nn  #
############################################

#%%
# load data
import pandas as pd

data_dir = '../data'
train_path = data_dir+"/book_rating_train.csv"
fea_name_path = data_dir+"/book_text_features_doc2vec/train_name_doc2vec100.csv"
fea_authors_path = data_dir+"/book_text_features_doc2vec/train_authors_doc2vec20.csv"
fea_desc_path = data_dir+"/book_text_features_doc2vec/train_desc_doc2vec100.csv"
# test_path = "data/book_rating_test.csv"
data = pd.read_csv(train_path, index_col = False, delimiter = ',', header=0)    # 23063
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

# # normalize numerical variables
# for feature in data:
#     scaler = StandardScaler()
#     scaler.fit(data[feature].values.reshape(-1,1))
#     data[feature] = scaler.transform(data[feature].values.reshape(-1,1))

# NOTE: first try to ignore the categorical variables
data = pd.concat([data, fea_name, fea_authors, fea_desc], axis=1)
print(data.shape)

#%%
# split data into training and validation sets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from skorch.helper import SliceDataset
import UOM.ML.NN.NeuralNet as my_nn

X = data.drop(columns=label_var).values
Y = data[label_var].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
trainData = my_nn.MyDataset(X_train, Y_train)
testData = my_nn.MyDataset(X_test, Y_test)

X_train = SliceDataset(trainData, idx=0)  # idx=0 is the default
y_train = SliceDataset(trainData, idx=1)


#%%
# define the model
from torch.utils.tensorboard import SummaryWriter, writer
from utils import save_checkpoint, load_checkpoint
import pickle

# training
torch.manual_seed(0)
num = 1
dir = "model"
wfile = "{}/model_{}".format(dir,num)
writer = SummaryWriter(f"model_{num}.log")
input_dim, output_dim = data.shape[1]-1, len(ind2label)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_dim = 70
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 1024
STEP_SIZE = 10  # 10
GAMMA = 0.5
PATIENCE = 500

#%%
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, Checkpoint, LRScheduler
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# define the model
model = NeuralNetClassifier(
    my_nn.Classifier,
    module__input_dim=input_dim,
    module__hidden_dim=hidden_dim,
    module__output_dim=output_dim,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    # lr=LR,
    batch_size=BATCH_SIZE,
    # max_epochs=EPOCHS,
    callbacks=[
        ('es', EarlyStopping(patience=PATIENCE)),
        ('cp', Checkpoint(dirname=dir, f_params='model_{}.pt'.format(num))),
        ('lr', LRScheduler(policy=StepLR, step_size=STEP_SIZE, gamma=GAMMA))
    ],
    device=device,
    # verbose=1,
)

#%%
from sklearn.model_selection import GridSearchCV

# define the grid search parameters
param_grid = {
    # 'batch_size': [64, 128, 256, 512, 1024],
    'max_epochs': [10, 50, 100, 200, 500],

    # 'module__hidden_dim': [40, 50, 60, 70, 80, 90, 100],
    # 'optimizer': [torch.optim.Adam, torch.optim.SGD],
    # 'optimizer_lr': [0.001, 0.01, 0.1, 0.2, 0.3],
    'lr': [0.001, 0.01, 0.1, 0.2, 0.3],
    # 'optimizer_weight_decay': [0.001, 0.01, 0.1, 0.2, 0.3],
    # 'lr_scheduler_step_size': [10, 20, 50],
    # 'lr_scheduler_gamma': [0.1, 0.2, 0.5]

}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)
 
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


#%%
# train the model
model.fit(trainData, y=None, epochs=EPOCHS, verbose=1)

# save the best model
best_model = my_nn.Classifier(input_dim, hidden_dim, output_dim)
load_checkpoint('{}/model_{}.pt'.format(dir,num), best_model)
file = open(wfile+'.mdl', 'wb')
pickle.dump(best_model, file)
file.close()
print("...........model_{} done!...........".format(num))


#%%

model = my_nn.Classifier(input_dim, hidden_dim, output_dim)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)



#%%



print("...........Start model_{}...........".format(num))
best_val_loss = float("Inf")

for epoch in range(EPOCHS):
    train_loss, train_count = 0, 0
    for i, (x, y) in enumerate(trainSet):
        x, y = x.to(device), y.to(device)

        train_count += 1
        model.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        train_loss += loss.detach().numpy()
        loss.backward()
        optimizer.step()

    val_loss, val_count = 0, 0
    with torch.no_grad():
        for x, y in valSet:
            x, y = x.to(device), y.to(device)
            val_count += 1
            output = model(x)
            val_loss += loss_fn(output, y).detach().numpy()
        cur_val_loss = val_loss / val_count

        # Record training and validation loss from each iter into the writer
        writer.add_scalar(f'model_{num}/Train/Loss', train_loss / train_count, epoch)
        writer.flush()
        writer.add_scalar(f'model_{num}/Validation/Loss', cur_val_loss, epoch)
        writer.flush()

        # checkpoint
        if best_val_loss > cur_val_loss:
            best_val_loss = cur_val_loss
            save_checkpoint('{}/Model_{}.pt'.format(dir,num), model, best_val_loss)
    print('Epoch: {}, TrainLoss: {:.3f}, ValLoss: {:.3f}'.format(epoch, 
                                train_loss / train_count, val_loss / val_count))
    
    scheduler.step()

# save the best model
best_model = my_nn.Classifier(input_dim, hidden_dim, output_dim)
load_checkpoint('{}/Model_{}.pt'.format(dir,0), best_model)
file = open(wfile+'.mdl', 'wb')
pickle.dump(best_model, file)
file.close()
print("...........model_{} done!...........".format(0))

#%%
# testing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

# load the best model
best_model = my_nn.Classifier(input_dim, hidden_dim, output_dim)
load_checkpoint('{}/Model_{}.pt'.format(dir,num), best_model)
y_true, y_pred = test(best_model, testSet)

# print the metrics
print("Accuracy: {:.3f}".format(accuracy_score(y_true, y_pred)))
print("Precision: {:.3f}".format(precision_score(y_true, y_pred, average='macro', zero_division=0)))
print("Recall: {:.3f}".format(recall_score(y_true, y_pred, average='macro')))
print("F1: {:.3f}".format(f1_score(y_true, y_pred, average='macro')))
print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()






# %%
