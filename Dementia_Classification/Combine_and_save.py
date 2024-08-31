import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from transformers import ViTFeatureExtractor, ViTForImageClassification
from sklearn.metrics import classification_report, confusion_matrix
import torch
import tensorflow as tf
from tensorflow import keras
import os
import shutil

# PyTorch ve TensorFlow'un GPU'yu kullanıp kullanmadığını kontrol edin
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for PyTorch: {device}")

print("Using device for TensorFlow:")
print("TensorFlow version:", tf.__version__)
print("GPU available: ", tf.config.list_physical_devices('GPU'))

# TensorFlow GPU bellek büyütmeyi devre dışı bırakma ve belirli bir limit ayarlama
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # GPU bellek büyütmeyi devre dışı bırak
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Alternatif olarak, belirli bir GPU bellek limiti belirleyin (MB cinsinden)
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])  # 4GB bellek limiti

    except RuntimeError as e:
        print(e)

# GPU belleğini temizleme
torch.cuda.empty_cache()

# Load the dataset
train_df = pd.read_parquet('../Datasets/Alzheimer_MRI_dataset_jpg/Data/train.parquet')
test_df = pd.read_parquet('../Datasets/Alzheimer_MRI_dataset_jpg/Data/test.parquet')

# Train ve test setlerini birleştir
combined_df = pd.concat([train_df, test_df], ignore_index=True)


def dict_to_image(image_dict):
    if isinstance(image_dict, dict) and 'bytes' in image_dict:
        byte_string = image_dict['bytes']
        nparr = np.frombuffer(byte_string, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    else:
        raise TypeError(f"Expected dictionary with 'bytes' key, got {type(image_dict)}")


# Apply the dict_to_image function
combined_df['image'] = combined_df['image'].apply(dict_to_image)


# Plotting function
def plotting(images, title="Images"):
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for img, ax in zip(images[:5], axes):
        ax.imshow(img)
        ax.axis('off')
    plt.suptitle(title)
    plt.show()


# Grayscale conversion and normalization büyük elmas
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (224, 224))  # Ensure consistent size
    normalized_image = cv2.normalize(resized_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.stack((normalized_image,) * 3, axis=-1)  # Convert grayscale to RGB


combined_df['processed_image'] = combined_df['image'].apply(preprocess_image)

plotting(combined_df['processed_image'], title="Processed Images")


# Brain tissue segmentation Hacim
def segment_brain_tissue(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return np.stack((thresholded,) * 3, axis=-1)  # Convert to RGB


combined_df['segmented_image'] = combined_df['processed_image'].apply(segment_brain_tissue)

plotting(combined_df['segmented_image'], title="Segmented Images")


# Edge detection for ventricle analysis Kenar tespit
def detect_edges(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray_image, 100, 200)
    return np.stack((edges,) * 3, axis=-1)  # Convert to RGB


combined_df['edges'] = combined_df['processed_image'].apply(detect_edges)

plotting(combined_df['edges'], title="Edge Detected Images")


# Klasör oluşturma ve var olanları temizleme
def create_and_clear_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        # Klasördeki mevcut dosyaları temizle
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


# Görüntüleri dosyalara kaydetme fonksiyonu
def save_images(df, base_directory):
    labels = df['label'].unique()
    for image_type in ['original', 'processed', 'segmented', 'edges']:
        for label in labels:
            label_dir = os.path.join(base_directory, image_type, str(label))
            create_and_clear_directory(label_dir)

            label_df = df[df['label'] == label]
            for i, (idx, row) in enumerate(label_df.iterrows()):
                if image_type == 'original':
                    img = row['image']
                elif image_type == 'processed':
                    img = row['processed_image']
                elif image_type == 'segmented':
                    img = row['segmented_image']
                elif image_type == 'edges':
                    img = row['edges']

                img_path = os.path.join(label_dir, f"{label}_{image_type}_{i + 1}.png")
                cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# Orijinal, işlenmiş, segment edilmiş ve kenar tespiti yapılmış görüntüleri kaydetme
save_images(combined_df, 'output/images')

print("Images have been saved successfully.")
