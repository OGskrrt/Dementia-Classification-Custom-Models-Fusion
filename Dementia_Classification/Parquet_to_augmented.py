import os
import cv2
import numpy as np
import pandas as pd

def dict_to_image(image_dict):
    if isinstance(image_dict, dict) and 'bytes' in image_dict:
        byte_string = image_dict['bytes']
        nparr = np.frombuffer(byte_string, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    else:
        raise TypeError(f"Expected dictionary with 'bytes' key, got {type(image_dict)}")

def save_images_from_parquet(parquet_file, output_base_folder):
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

    # Parquet dosyasını oku
    df = pd.read_parquet(parquet_file)

    # Etiketlerin sayısını kontrol et
    label_counts = df['label'].value_counts()
    print("Initial label counts:")
    print(label_counts)

    # Tüm unique etiketler (label) için döngü
    for label in df['label'].unique():
        # Etikete göre klasör oluştur
        label_folder = os.path.join(output_base_folder, f'label_{label}')
        os.makedirs(label_folder, exist_ok=True)

        # Etikete sahip tüm görüntüler
        images = df[df['label'] == label]['image'].tolist()

        for idx, image_dict in enumerate(images):
            img = dict_to_image(image_dict)

            if img is not None:
                # Görüntüyü dosyaya kaydet
                image_path = os.path.join(label_folder, f'image_{idx}.png')
                cv2.imwrite(image_path, img)
                print(f"Saved: {image_path}")
            else:
                print(f"Could not decode image at index {idx} for label {label}")

def apply_augmentation(image, method_index):
    if method_index == 0:
        angle = np.random.uniform(-15, 15)  # 30 yerine 15 derece
        augmented_image = rotate_image(image, angle)
    elif method_index == 1:
        height, width = image.shape[:2]
        tx = np.random.uniform(-0.1 * width, 0.1 * width)  # Daha geniş bir kaydırma aralığı
        ty = np.random.uniform(-0.1 * height, 0.1 * height)  # Daha geniş bir kaydırma aralığı
        augmented_image = shift_image(image, tx, ty)
    elif method_index == 2:
        augmented_image = flip_image(image, horizontal=True)
    elif method_index == 3:
        augmented_image = flip_image(image, horizontal=False)
    else:
        augmented_image = image
    return augmented_image

def rotate_image(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def shift_image(image, dx, dy):
    height, width = image.shape[:2]
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_image = cv2.warpAffine(image, translation_matrix, (width, height))
    return shifted_image

def flip_image(image, horizontal=True):
    if horizontal:
        return cv2.flip(image, 1)
    else:
        return cv2.flip(image, 0)

def crop_black_borders(image):
    # Grayscale'e çevir
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Binary threshold ile siyah ve beyaz yap
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    # Non-zero (siyah olmayan) piksellerin koordinatlarını al
    coords = cv2.findNonZero(thresh)
    # Siyah olmayan piksellerin etrafında dikdörtgen bir bölge belirle
    x, y, w, h = cv2.boundingRect(coords)
    # Bu bölgeyi kırp
    cropped = image[y:y+h, x:x+w]
    return cropped

def augment_images_in_folder(input_folder, output_folder, target_count):
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Giriş klasörü bulunamadı: {input_folder}")

    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    current_count = len(image_files)

    augment_count = target_count - current_count
    iteration = 0
    while augment_count > 0:
        iteration += 1
        for image_file in image_files:
            if augment_count <= 0:
                break
            image_path = os.path.join(input_folder, image_file)
            mr_image = cv2.imread(image_path)

            if mr_image is None:
                print(f"Görüntü yüklenemedi: {image_path}")
                continue

            mr_image_resized = cv2.resize(mr_image, (128, 128))

            if len(mr_image_resized.shape) == 3 and mr_image_resized.shape[2] == 3:
                mr_image_gray = cv2.cvtColor(mr_image_resized, cv2.COLOR_BGR2GRAY)
            else:
                mr_image_gray = mr_image_resized

            mr_image_normalized = mr_image_gray / 255.0

            for i in range(4):
                if augment_count <= 0:
                    break
                augmented_image = apply_augmentation(mr_image_normalized, (i + iteration) % 4)
                augmented_image_uint8 = (augmented_image * 255).astype(np.uint8)
                cropped_image = crop_black_borders(augmented_image_uint8)
                output_path = os.path.join(output_folder,
                                           f'{os.path.splitext(image_file)[0]}_augmented_{current_count + iteration}_{i}.png')
                cv2.imwrite(output_path, cropped_image)
                augment_count -= 1

            print(f"İşlenen görüntüler kaydedildi: {image_file}")

if __name__ == '__main__':
    parquet_file = '../Datasets/Alzheimer_MRI_dataset_jpg/Data/train.parquet'  # Parquet dosyasının yolu
    original_output_folder = 'Parquet/Original'  # Orijinal görüntülerin kayıt yolu
    augmented_output_folder = 'Parquet/Augmented'  # Artırılmış görüntülerin kayıt yolu

    # Orijinal görüntüleri kaydet
    save_images_from_parquet(parquet_file, original_output_folder)

    # Etiketlerin sayısını kontrol et
    label_counts = {}
    for label in os.listdir(original_output_folder):
        label_folder = os.path.join(original_output_folder, label)
        label_counts[label] = len(os.listdir(label_folder))

    print("Initial label counts:")
    print(label_counts)

    # En yüksek etiket sayısını belirle
    target_count = label_counts['label_2']  # Hedef sayıyı label_2'ye eşitle

    print(f"Target count per label: {target_count}")

    # Orijinal görüntüleri artır ve dengeli hale getir
    for label in os.listdir(original_output_folder):
        if label == 'label_2':  # label_2'yi atla
            continue
        input_folder = os.path.join(original_output_folder, label)
        output_folder = os.path.join(augmented_output_folder, label)
        augment_images_in_folder(input_folder, output_folder, target_count)

    # Orijinal label_2'yi augmented klasörüne kopyala
    label_2_input_folder = os.path.join(original_output_folder, 'label_2')
    label_2_output_folder = os.path.join(augmented_output_folder, 'label_2')
    os.makedirs(label_2_output_folder, exist_ok=True)
    for image_file in os.listdir(label_2_input_folder):
        src_path = os.path.join(label_2_input_folder, image_file)
        dst_path = os.path.join(label_2_output_folder, image_file)
        img = cv2.imread(src_path)
        cropped_image = crop_black_borders(img)
        cv2.imwrite(dst_path, cropped_image)

    # Tekrar label_3 artırma işlemi yap
    label_3_folder = os.path.join(augmented_output_folder, 'label_3')
    while len(os.listdir(label_3_folder)) < target_count:
        augment_images_in_folder(label_3_folder, label_3_folder, target_count)

    # Sonuçları kontrol et
    final_label_counts = {}
    for label in os.listdir(augmented_output_folder):
        label_folder = os.path.join(augmented_output_folder, label)
        final_label_counts[label] = len(os.listdir(label_folder))
    print("Final label counts after augmentation:")
    print(final_label_counts)
