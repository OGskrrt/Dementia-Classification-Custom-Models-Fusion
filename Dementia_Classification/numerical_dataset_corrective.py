import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from shutil import copyfile

def find_image(image_name, root_folder):
    for root, dirs, files in os.walk(root_folder):
        if image_name in files:
            return os.path.join(root, image_name)
    return None

def preprocess_data(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

    imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    return df

def CreateRows(image_folder, csv_file):
    df = pd.read_csv(csv_file)
    #df = preprocess_data(df)
    images = []
    labels = []
    features = []
    ids = []

    for index, row in df.iterrows():
        image_name = row['name']
        image_path = find_image(image_name, image_folder)

        if image_path:
            images.append(image_path)
        else:
            print(f"Image {image_name} not found in {image_folder}")

        labels.append(row['label'])
        features.append(row.drop(['name', 'label']).to_dict())
        ids.append(row['name'])

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return images, labels, features, ids

def save_images_and_dataset(images, labels, features, ids, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)

    image_paths = []
    for i, img_path in enumerate(images):
        img_name = os.path.basename(img_path)
        new_img_path = os.path.join(image_dir, img_name)
        copyfile(img_path, new_img_path)
        image_paths.append(new_img_path)

    data = {
        'image_path': image_paths,
        'label': labels,
        'id': ids
    }
    data.update(pd.DataFrame(features).to_dict(orient='list'))

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, 'dataset.csv'), index=False)

def main():
    data_dirCNN = 'Output_filtered/processed'
    data_dirANN = 'cont_drop_rlm.csv'

    images, labels, features, ids = CreateRows(data_dirCNN, data_dirANN)

    # Save dataset locally
    save_images_and_dataset(images, labels, features, ids, 'Fusion_deneme_setleri/Fusion_sets/vk5')



if __name__ == "__main__":
    main()
