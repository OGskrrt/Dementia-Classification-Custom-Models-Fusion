import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    jaccard_score, roc_auc_score, mean_squared_error, confusion_matrix
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from PIL import Image
import time
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def crop_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image

# Model tanımlamaları
class TunedCNN(nn.Module):
    def __init__(self):
        super(TunedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 56 * 56, 256)
        self.batchnorm_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 4)
        self.batchnorm_fc2 = nn.BatchNorm1d(4)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        x = F.mish(self.batchnorm1(self.conv1(x)))
        x = self.pool(x)

        x = F.mish(self.batchnorm2(self.conv2(x)))
        x = self.pool(x)

        x = self.flatten(x)

        x = F.mish(self.batchnorm_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = F.mish(self.batchnorm_fc2(self.fc2(x)))
        x = self.dropout_fc(x)

        return x


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


class FusionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.data = self.preprocess_data(self.data)

    def __len__(self):
        return len(self.data)

    def preprocess_data(self, df):
        cols = [col for col in df.columns if col not in ["label", "id", "image_path"]]

        # Eksik verileri doldurmak için SimpleImputer kullanımı
        #imputer = SimpleImputer(strategy='median')
        #df[cols] = imputer.fit_transform(df[cols])

        # StandardScaler kullanımı
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])

        return df



    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        image = cv2.imread(image_path)  # Convert to grayscale


        image = crop_black_borders(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.transform:
              # Convert back to PIL Image for transformations
            image = self.transform(image)

         # Convert PIL Image to numpy array
        label = row['label']
        features = row.drop(['image_path', 'label', 'id']).values.astype(np.float32)
        return image, features, label



class FUSION(nn.Module):
    def __init__(self):
        super(FUSION, self).__init__()
        self.fc0 = nn.Linear(8, 4)

    def forward(self, x):
        x = torch.sigmoid(self.fc0(x))
        return x


def load_model(model_path, model_class, input_size=None):
    if input_size is not None:
        model = model_class(input_size)
    else:
        model = model_class()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def calculate_metrics(y_true, y_pred, y_scores, num_classes):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    iou = jaccard_score(y_true, y_pred, average='weighted')
    y_scores = np.exp(y_scores) / np.sum(np.exp(y_scores), axis=1, keepdims=True)  # Apply softmax
    auc = roc_auc_score(y_true, y_scores, multi_class='ovr')
    mse = mean_squared_error(y_true, y_scores.argmax(axis=1))
    rmse = np.sqrt(mse)
    return accuracy, precision, recall, f1, iou, auc, mse, rmse


def save_best_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Best model saved to {path}")


def train_fusion_model(cnn_model, ann_model, fusion_model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, patience=3):
    best_loss = float('inf')
    patience_counter = 0
    start_time = time.time()  # Start time for training
    scaler = GradScaler()  # Initialize GradScaler for mixed precision

    for epoch in range(num_epochs):
        fusion_model.train()
        running_loss = 0.0

        for images, features, labels in train_loader:
            images, features, labels = images.to(device), features.to(device), labels.to(device)

            with torch.no_grad():
                cnn_output = cnn_model(images)
                ann_output = ann_model(features)

            cnn_output = cnn_output.view(cnn_output.size(0), -1) * 1.0
            ann_output = ann_output.view(ann_output.size(0), -1) * 1.0

            combined_output = torch.cat((cnn_output, ann_output), dim=1)

            optimizer.zero_grad()
            with autocast():
                outputs = fusion_model(combined_output)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Validation loss calculation and printing every 5 epochs
        if (epoch + 1) % 6 == 0:
            fusion_model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for val_images, val_features, val_labels in val_loader:
                    val_images, val_features, val_labels = val_images.to(device), val_features.to(device), val_labels.to(device)
                    val_cnn_output = cnn_model(val_images)
                    val_ann_output = ann_model(val_features)

                    val_cnn_output = val_cnn_output.view(val_cnn_output.size(0), -1) * 1.0
                    val_ann_output = val_ann_output.view(val_ann_output.size(0), -1) * 1.0

                    val_combined_output = torch.cat((val_cnn_output, val_ann_output), dim=1)

                    val_outputs = fusion_model(val_combined_output)
                    val_loss = criterion(val_outputs, val_labels)
                    val_running_loss += val_loss.item()

            val_loss = val_running_loss / len(val_loader)
            print(f'Validation Loss after Epoch [{epoch + 1}/{num_epochs}]: {val_loss:.4f}')

        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            save_best_model(fusion_model, "Fusion_deneme_setleri/modeller/Fusion_vk5.pt")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

    end_time = time.time()  # End time for training
    training_time = end_time - start_time
    print(f'Training completed in {training_time:.2f} seconds')


def evaluate_model(cnn_model, ann_model, fusion_model, data_loader, mode="test"):
    fusion_model.eval()
    all_labels = []
    all_preds = []
    all_scores = []
    start_time = time.time()  # Start time for evaluation

    with torch.no_grad():
        for images, features, labels in data_loader:
            images, features, labels = images.to(device), features.to(device), labels.to(device)
            cnn_output = cnn_model(images)
            ann_output = ann_model(features)

            cnn_output = cnn_output.view(cnn_output.size(0), -1) * 1.0
            ann_output = ann_output.view(ann_output.size(0), -1) * 1.0

            combined_input = torch.cat((cnn_output, ann_output), dim=1)
            outputs = fusion_model(combined_input)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())

    end_time = time.time()  # End time for evaluation
    evaluation_time = end_time - start_time
    print(f'{mode.capitalize()} evaluation completed in {evaluation_time:.2f} seconds')

    metrics = calculate_metrics(all_labels, all_preds, np.array(all_scores), num_classes=4)

    print(f"{mode.capitalize()} Metrics:")
    print(f"Accuracy: {metrics[0]:.4f}")
    print(f"Precision: {metrics[1]:.4f}")
    print(f"Recall: {metrics[2]:.4f}")
    print(f"F1 Score: {metrics[3]:.4f}")
    print(f"IoU: {metrics[4]:.4f}")
    print(f"AUC: {metrics[5]:.4f}")
    print(f"MSE: {metrics[6]:.4f}")
    print(f"RMSE: {metrics[7]:.4f}")
    print(classification_report(all_labels, all_preds, digits=4))
    print(f"Confusion Matrix:\n{confusion_matrix(all_labels, all_preds)}")

    return metrics



def main():
    dataset_path = 'Fusion_deneme_setleri/Fusion_sets/vk4'
    cnn_model_path = 'Fusion_deneme_setleri/modeller/CNNvk5.pt'
    ann_model_path = 'Fusion_deneme_setleri/modeller/ANN_vk5.pt'

    test_size = 0.2
    val_size = 0.1
    batch_size = 128

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    dataset = FusionDataset(os.path.join(dataset_path, 'dataset.csv'), transform=transform)
    train_size = int((1 - test_size - val_size) * len(dataset))
    val_size = int(val_size * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    print(f"Feature size for ANN model: {train_loader.dataset[0][1].shape[0]}")

    cnn_model = load_model(cnn_model_path, TunedCNN)
    ann_model = load_model(ann_model_path, ANNmodel, input_size=train_loader.dataset[0][1].shape[0])

    fusion_model = FUSION().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(fusion_model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    train_fusion_model(cnn_model, ann_model, fusion_model, train_loader, val_loader, criterion, optimizer, scheduler,
                       num_epochs=30, patience=5)

    print("\nFinal Evaluation on Train Set")
    train_metrics = evaluate_model(cnn_model, ann_model, fusion_model, train_loader, mode="train")

    print("\nFinal Evaluation on Test Set")
    test_metrics = evaluate_model(cnn_model, ann_model, fusion_model, test_loader, mode="test")


if __name__ == '__main__':
    main()



