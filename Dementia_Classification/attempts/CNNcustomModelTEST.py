import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, jaccard_score, roc_auc_score, mean_squared_error, roc_curve, auc
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

        #image = crop_black_borders(image)

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

def load_model(model_path, model):
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_outputs)

def calculate_metrics(y_true, y_pred, y_scores):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    iou = jaccard_score(y_true, y_pred, average='weighted')
    y_scores = np.exp(y_scores) / np.sum(np.exp(y_scores), axis=1, keepdims=True)  # Apply softmax
    auc_score = roc_auc_score(y_true, y_scores, multi_class='ovr')
    mse = mean_squared_error(y_true, y_scores.argmax(axis=1))
    rmse = np.sqrt(mse)
    return accuracy, precision, recall, f1, iou, auc_score, mse, rmse

def plot_roc_curve(y_true, y_scores, n_classes, title):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def print_metrics(y_true, y_pred, y_scores, n_classes):
    accuracy, precision, recall, f1, iou, auc_score, mse, rmse = calculate_metrics(y_true, y_pred, y_scores)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"AUC: {auc_score:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Classification Report:\n{classification_report(y_true, y_pred, digits=4)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

def main():
    data_dir = 'Fusion_deneme_setleri/vk4/processed_kirpilmamis'
    model_path = 'Fusion_deneme_setleri/modeller/CNNvk4.pt'

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    dataset = CustomDataset(data_dir, transform=data_transforms)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TunedCNN().to(device)

    # Load and evaluate on training data
    model = load_model(model_path, model)
    y_train_true, y_train_pred, y_train_scores = evaluate_model(model, train_loader, device)
    print("Training Metrics:")
    print_metrics(y_train_true, y_train_pred, y_train_scores, n_classes=4)


    # Evaluate on test data
    y_test_true, y_test_pred, y_test_scores = evaluate_model(model, test_loader, device)
    print("")
    print("")
    print("")
    print("")
    print("Test Metrics:")
    print_metrics(y_test_true, y_test_pred, y_test_scores, n_classes=4)

    plot_roc_curve(y_train_true, y_train_scores, n_classes=4, title='Training ROC Curve')
    plot_roc_curve(y_test_true, y_test_scores, n_classes=4, title='Test ROC Curve')

    # Compare training and test AUC scores for overfitting
    _, _, _, _, _, train_auc, _, _ = calculate_metrics(y_train_true, y_train_pred, y_train_scores)
    _, _, _, _, _, test_auc, _, _ = calculate_metrics(y_test_true, y_test_pred, y_test_scores)

    if train_auc > test_auc + 0.1:
        print(f"Possible overfitting detected: Train AUC ({train_auc:.4f}) > Test AUC ({test_auc:.4f})")

if __name__ == '__main__':
    main()
