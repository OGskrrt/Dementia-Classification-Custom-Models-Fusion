import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, jaccard_score, roc_auc_score, mean_squared_error
import time
import torch.nn.functional as F
from datetime import datetime

torch.cuda.empty_cache()


def crop_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image


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

        image = crop_black_borders(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.transform:
            image = self.transform(image)
        return image, label


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

# Load model and print state_dict keys for debugging
def load_model_and_print_keys(model_path, model):
    state_dict = torch.load(model_path)
    print("Keys in loaded state_dict:")
    for key in state_dict.keys():
        print(key)

    model_state_dict = model.state_dict()
    print("\nKeys in model's state_dict:")
    for key in model_state_dict.keys():
        print(key)

    return state_dict

def main():
    data_dir = 'Output_filtered/processed'

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

    dataset = CustomDataset(data_dir, transform=data_transforms['train'])
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    num_classes = len(set(dataset.labels))

    def create_model():
        model = TunedCNN()
        return model.to(device)

    criterion = nn.CrossEntropyLoss()

    def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=50, patience=3):
        best_loss = float('inf')
        patience_counter = 0

        model.train()
        start_time = time.time()

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

            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                save_best_model(model, 'Github/attempts/raw_models_for_vk5/best_model_ManuelModel.pt')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping")
                break

        end_time = time.time()
        print(f'Training Time: {end_time - start_time:.2f} seconds')

        model.load_state_dict(torch.load('Github/attempts/raw_models_for_vk5/best_model_ManuelModel.pt'))

        return model

    def evaluate_model(model, dataloaders, phase):
        model.eval()
        all_preds = []
        all_labels = []
        all_outputs = []
        start_time = time.time()
        with torch.no_grad():
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
        end_time = time.time()
        print(f'{phase.capitalize()} Evaluation Time: {end_time - start_time:.2f} seconds')

        return all_labels, all_preds, all_outputs

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

    def save_results(model_path, y_true, y_pred, dataset_path):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report = classification_report(y_true, y_pred)
        confusion = confusion_matrix(y_true, y_pred)

        results_path = model_path.replace('.pt', '_results.txt')
        with open(results_path, 'w') as f:
            f.write(f"Model Path: {model_path}\n")
            f.write(f"Dataset Path: {dataset_path}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(np.array2string(confusion))

        print(f"Results saved to {results_path}")

    def train_and_evaluate(dataloaders):
        model = create_model()
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=50, patience=3)

        for phase in ['train', 'val']:
            y_true, y_pred, y_scores = evaluate_model(model, dataloaders, phase)
            accuracy, precision, recall, f1, iou, auc, mse, rmse = calculate_metrics(y_true, y_pred, y_scores, num_classes)

            print(f"{phase.capitalize()} Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"Dice (F1-score): {f1:.4f}")
            print(f"IoU (Jaccard): {iou:.4f}")
            print(f"AUC: {auc:.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"Classification Report:\n{classification_report(y_true, y_pred)}")
            print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

            if phase == 'val':
                save_results('Github/attempts/raw_models_for_vk5/best_model_ManuelModel.pt', y_true, y_pred, data_dir)

    dataloaders_augmented = {'train': train_loader, 'val': test_loader}
    print("Augmented Images:")
    train_and_evaluate(dataloaders_augmented)


if __name__ == '__main__':
    main()
