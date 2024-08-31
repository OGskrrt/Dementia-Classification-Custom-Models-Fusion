import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

torch.cuda.empty_cache()


def crop_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image


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


def predict_single_image(image_path, model, transform, device):
    image = cv2.imread(image_path)
    image = crop_black_borders(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if transform:
        image = transform(image)
    image = image.unsqueeze(0).to(device)  # Batch dimension ekleniyor

    model.eval()
    with torch.no_grad():
        output = model(image)
        print(output[0].cpu().numpy())
        output2 = output.view(output.size(0), -1)
        print(output2)
        _, prediction = torch.max(output, 1)

    return prediction.item()


def main():
    model_path = 'Github/attempts/raw_models_for_vk5/best_model_ManuelModel.pt'
    single_image_path = 'Output_filtered/processed/0/0_processed_426_augmented_3.png' # Buraya tahmin etmek istediğiniz görüntünün yolunu yazın

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TunedCNN().to(device)
    model = load_model(model_path, model)

    predicted_label = predict_single_image(single_image_path, model, data_transforms, device)
    print(f"The predicted label for the image is: {predicted_label}")


if __name__ == '__main__':
    main()
