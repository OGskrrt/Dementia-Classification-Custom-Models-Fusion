import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.preprocessing import StandardScaler
import time

torch.cuda.empty_cache()

# Custom dataset class
class AlzheimerDataset(Dataset):
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
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    data_dir = '../Parquet/Augmented'  # Path to augmented images
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # Transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create datasets and dataloaders
    train_dataset = AlzheimerDataset(train_dir, transform=data_transforms['train'])
    test_dataset = AlzheimerDataset(test_dir, transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Define the model
    class VGG16Model(nn.Module):
        def __init__(self, num_classes):
            super(VGG16Model, self).__init__()
            self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.model.classifier[6] = nn.Linear(4096, num_classes)

        def forward(self, x):
            return self.model(x)

    # Initialize the model, loss function, and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = len(set(train_dataset.labels))

    def create_model():
        model = VGG16Model(num_classes=num_classes)
        return model.to(device)

    criterion = nn.CrossEntropyLoss()

    def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
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
        end_time = time.time()  # End time
        print(f'Training Time: {end_time - start_time:.2f} seconds')

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
        model = create_model()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        model = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)
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
