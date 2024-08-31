import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import time
import torch.nn.functional as F
from torchvision import datasets, models

torch.cuda.empty_cache()

def crop_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

# Özel veri seti sınıfı tanımlama
class CustomDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = []
        self.labels = []

        for label_folder in os.listdir(image_folder):
            label_path = os.path.join(image_folder, label_folder)
            if os.path.isdir(label_path):
                label = int(label_folder.split('_')[-1])
                for image_file in os.listdir(label_path):
                    self.images.append(os.path.join(label_path, image_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)

        # Görüntüyü kırpma fonksiyonunu kullanarak kırp
        image = crop_black_borders(image)

        # Gri tonlamaya çevir ve dönüşümleri uygula
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    data_dir = '../Output_filtered/processed'  # Path to data

    # Transformations with Data Augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ]),
    }

    # Create datasets and dataloaders
    dataset = CustomDataset(data_dir, transform=data_transforms['train'])
    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Define the model
    class CustomCNN(nn.Module):
        def __init__(self):
            super(CustomCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.batchnorm1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.batchnorm2 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.batchnorm3 = nn.BatchNorm2d(128)
            self.pool = nn.MaxPool2d(2, 2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(128 * 28 * 28, 256)
            self.batchnorm_fc1 = nn.BatchNorm1d(256)
            self.fc2 = nn.Linear(256, 128)
            self.batchnorm_fc2 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128, 4)  # 6 sınıf

        def forward(self, x):
            x = F.mish(self.batchnorm1(self.conv1(x)))
            x = self.pool(x)
            x = F.mish(self.batchnorm2(self.conv2(x)))
            x = self.pool(x)
            x = F.mish(self.batchnorm3(self.conv3(x)))
            x = self.pool(x)
            x = self.flatten(x)
            x = F.mish(self.batchnorm_fc1(self.fc1(x)))
            x = F.dropout(x, 0.5)
            x = F.mish(self.batchnorm_fc2(self.fc2(x)))
            x = F.dropout(x, 0.5)
            x = self.fc3(x)
            return x


    class TunedCNN2(nn.Module):
        def __init__(self):
            super(TunedCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.batchnorm1 = nn.BatchNorm2d(num_features=32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * 56 * 56, 128)
            self.drop1 = nn.Dropout(p=0.2)
            self.out = nn.Linear(128, 4)  # 4 sınıf

        def forward(self, x):
            x = F.mish(self.conv1(x))
            x = self.pool1(x)
            x = self.batchnorm1(x)
            x = F.mish(self.conv2(x))
            x = self.pool2(x)
            x = self.flatten(x)
            x = self.fc1(x)
            leaky = nn.LeakyReLU(0.01)
            x = leaky(x)
            x = self.drop1(x)
            x = self.out(x)
            return x
    # Initialize the model, loss function, and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model = CustomCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    def train_model(model, dataloaders, criterion, optimizer, num_epochs=100, patience=4):
        best_loss = float('inf')
        patience_counter = 0

        model.train()
        start_time = time.time()  # Start time

        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in dataloaders['train']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders['train'].dataset)
            print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

            # Early Stopping
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in dataloaders['val']:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)

            val_loss /= len(dataloaders['val'].dataset)
            print(f'Validation Loss: {val_loss:.4f}')

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), '../best_model.pt')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping")
                break

        end_time = time.time()  # End time
        print(f'Training Time: {end_time - start_time:.2f} seconds')

        # Load the best model
        model.load_state_dict(torch.load('../best_model.pt'))

        return model

    def evaluate_model(model, dataloaders):
        model.eval()
        all_preds = []
        all_labels = []
        start_time = time.time()  # Start time
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        end_time = time.time()  # End time
        print(f'Evaluation Time: {end_time - start_time:.2f} seconds')

        return all_labels, all_preds

    def calculate_metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        iou = jaccard_score(y_true, y_pred, average='weighted')
        return accuracy, precision, recall, f1, iou

    # Training and evaluating function
    def train_and_evaluate(dataloaders):
        model = CustomCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        model = train_model(model, dataloaders, criterion, optimizer, num_epochs=200, patience=50)
        y_test, y_pred = evaluate_model(model, dataloaders)

        accuracy, precision, recall, f1, iou = calculate_metrics(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Dice (F1-score): {f1:.4f}")
        print(f"IoU (Jaccard): {iou:.4f}")

        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Training and evaluating for augmented images
    dataloaders_augmented = {'train': train_loader, 'val': test_loader}
    print("Augmented Images:")
    train_and_evaluate(dataloaders_augmented)


if __name__ == '__main__':
    main()
