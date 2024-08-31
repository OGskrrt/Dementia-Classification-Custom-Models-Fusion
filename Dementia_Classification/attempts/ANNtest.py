import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, jaccard_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
import torch.nn.functional as F
import time

torch.cuda.empty_cache()

# Veri yükleme ve ön işleme
df = pd.read_csv('cont_drop_rlm.csv')
df = df.drop('name', axis=1)

cols = [col for col in df.columns if "label" not in col]

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#for col in cols:
    #if check_outlier(df, col):
        #replace_with_thresholds(df, col)

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

X = df.drop('label', axis=1).values
y = df['label'].values

def preprocess_data(X):
    imputer = SimpleImputer(strategy='median')
    X_processed = imputer.fit_transform(X)
    return X_processed

#X_processed = preprocess_data(X)

label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.17, random_state=123)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

inputsize = X_train.shape[1]

# Model tanımlaması
class ANNmodel(nn.Module):
    def __init__(self, inputsize):
        super(ANNmodel, self).__init__()
        self.ln1 = nn.Linear(inputsize, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.ln2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.ln3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.ln4 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.ln5 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.ln6 = nn.Linear(256, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.ln7 = nn.Linear(128, 64)
        self.bn7 = nn.BatchNorm1d(64)
        self.ln8 = nn.Linear(64, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.ln1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.ln2(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.ln3(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn4(self.ln4(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn5(self.ln5(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn6(self.ln6(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn7(self.ln7(x)), negative_slope=0.01)
        x = torch.sigmoid(self.ln8(x))
        return x

def save_best_model(model, path):
    torch.save(model.state_dict(), path)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, patience=10):
    best_loss = float('inf')
    patience_counter = 0
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        print(f'Validation Loss: {val_loss:.4f}')
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            save_best_model(model, 'ANN_deneme.pt')
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break
    end_time = time.time()
    print(f'Training Time: {end_time - start_time:.2f} seconds')
    model.load_state_dict(torch.load('ANN_deneme.pt'))
    return model

def evaluate_model(model, dataloader, phase='test'):
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(F.softmax(outputs, dim=1).cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    print(f"Classification Report ({phase}):")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"Confusion Matrix ({phase}):")
    print(confusion_matrix(y_true, y_pred))
    print(f"Accuracy Score ({phase}):", accuracy_score(y_true, y_pred))
    print(f"Precision Score ({phase}):", precision_score(y_true, y_pred, average='weighted'))
    print(f"Recall Score ({phase}):", recall_score(y_true, y_pred, average='weighted'))
    print(f"F1 Score ({phase}):", f1_score(y_true, y_pred, average='weighted'))
    print(f"Jaccard Score ({phase}):", jaccard_score(y_true, y_pred, average='weighted'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = ANNmodel(inputsize).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=1000)


"""
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5, weight_decay=0.001)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=1500, anneal_strategy='cos')

scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.006, step_size_up=2000)

scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0001, total_steps=1500, anneal_strategy='cos')

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.2, weight_decay=0.001)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=1500, anneal_strategy='cos')

-2
optimizer = optim.AdamW(model.parameters(), lr=0.0005)
scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.006, step_size_up=2000)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

-1
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=500, anneal_strategy='cos')

-3
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.01)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=500, anneal_strategy='cos')

"""

model = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=500, patience=50)


print("\nEvaluation on Train Set")
evaluate_model(model, train_loader, phase='train')

print("\nEvaluation on Test Set")
evaluate_model(model, test_loader, phase='test')
