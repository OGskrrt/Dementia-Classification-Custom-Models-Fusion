# Dementia Classification from MRI Images Using Machine Learning and Deep Learning Models

## **Note:** For a detailed report prepared in Turkish, please refer to the `Rapor.pdf` file. This document provides an in-depth analysis and comprehensive explanation of the methods, results, and findings discussed in this project.

**Important:** The `.pt` files (model weights) for the CNN models are not included in this repository due to their large size. To use the models, you will need to train them using the provided scripts or download the weights separately if available.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Approaching](#model-approaching)
5. [Initial Results](#initial-results)
6. [Extracting Numerical Data](#extracting-numerical-data)
7. [ANN Model Development](#ann-model-development)
8. [ANN and CNN Fusion Model](#ann-and-cnn-fusion-model)
9. [Final Results](#final-results)
10. [Conclusion](#conclusion)
11. [Future Work](#future-work)
12. [How to Use](#how-to-use)

## Introduction

Alzheimer’s disease is a major cause of dementia, affecting millions of people worldwide, particularly older adults. The disease causes memory loss, confusion, and difficulties with daily tasks. Early and accurate detection of dementia levels is crucial for effective management and treatment.

This project aims to improve the classification accuracy of dementia levels using MRI images. The current best performance on this dataset, reported on public platforms, is an accuracy of 98.7%. Our goal is to surpass this benchmark by developing custom models and a fusion approach that combines deep learning (CNN) and traditional machine learning (ANN) techniques.

## Dataset Description

The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/datasets/borhanitrash/alzheimer-mri-disease-classification-dataset) and consists of 6400 MRI images categorized into four classes: Mild Demented, Moderate Demented, Non-Demented, and Very Mild Demented. Initially available in Parquet format, the data was extracted, processed, and augmented using custom scripts.

Key challenges include class imbalance and the need for various data manipulations to improve model performance.

## Data Preprocessing

Data preprocessing involved extensive exploration, augmentation, and manipulation of MRI images:
- **Data Augmentation:** Techniques such as rotation, shifting, brightness adjustment, zooming, and noise addition were applied to create a balanced and diverse dataset.
- **Medical Insights:** Images were enhanced to focus on brain regions critical for diagnosing Alzheimer’s, such as the hippocampus and areas showing signs of atrophy.

Five different datasets were prepared, each with varying levels of manipulation, to explore the most effective preprocessing strategies.

## Model Approaching

We initially tested several pre-trained CNN models, including VGG16, AlexNet, Unet, ResNet34, and DenseNet121. Despite their strengths in other classification tasks, these models did not perform well on our specific dataset. Therefore, we developed a custom CNN model tailored to the unique characteristics of our data.

### Custom CNN Model Development
The custom CNN model was specifically designed to address challenges such as class imbalance and subtle differences in dementia levels. It included advanced components like convolutional layers, pooling layers, and dropout regularization to enhance performance and prevent overfitting.

### Transfer Learning Attempts
Feature extraction using pre-trained models like AlexNet and VGG16 was also explored but did not yield satisfactory results compared to the custom CNN approach.

## Initial Results

The custom CNN model achieved promising results, significantly outperforming pre-trained models. The model demonstrated its strength, particularly in distinguishing harder-to-classify categories such as Moderate Demented.

## Extracting Numerical Data

Beyond visual features, numerical data was extracted from the MRI images, including brain volume, gray matter volume, and intensity measures. These features were compiled into a structured dataset, providing additional quantitative insights for classification.

## ANN Model Development

A custom ANN model was developed to handle the numerical data, as pre-trained machine learning models like XGBoost, LightGBM, and RandomForest were insufficient. The ANN model effectively processed these features, complementing the CNN’s visual analysis.

### Training Strategies
Advanced training techniques such as cyclic learning rates, dropout regularization, and batch normalization were employed to enhance the ANN model’s performance, ensuring rapid convergence and robustness.

## ANN and CNN Fusion Model

### Concept and Implementation
The hybrid fusion model combined the strengths of the custom CNN and ANN models. By merging image data with numerical features, the fusion model achieved superior performance compared to either model alone.

### Fusion Strategy
Various fusion strategies were tested, including 75% CNN - 25% ANN and 50% CNN - 50% ANN, with the latter proving most effective. This approach highlighted the power of integrating diverse data types for complex medical classification tasks.

## Final Results

The fusion model set a new benchmark in dementia classification, achieving record accuracy and F1-scores, particularly in challenging categories. The balanced fusion strategy provided the most reliable and robust predictions.

## Conclusion

This project demonstrated that integrating deep learning with traditional machine learning significantly improves dementia classification from MRI images. The fusion model, combining CNN and ANN, surpassed previous benchmarks and provided a comprehensive, clinically relevant diagnostic tool.

## Future Work

Future research directions include:
- **Expanding the Dataset:** Testing on larger, more diverse datasets to enhance generalizability.
- **Exploring Advanced Fusion Techniques:** Investigating new fusion strategies, such as neural attention mechanisms, for further optimization.
- **Clinical Integration:** Adapting the model for real-world use in medical settings to support doctors in diagnosing and managing Alzheimer’s disease.

## How to Use

### Prerequisites
- Python 3.x
- Required libraries: PyTorch, OpenCV, Pandas, Scikit-learn, etc.

### Scripts
- `Parquet_to_augmented.py`: Converts Parquet data into images and organizes them into labeled folders, preparing the dataset for further processing.
- `Combine_and_save.py`: Applies specific manipulations to images to enhance focus on critical brain areas affected by dementia, such as the hippocampus.
- `osman.ipynb`: Contains code for data augmentation processes, including rotation, shifting, and other techniques to create a balanced dataset.
- `transferLearning.py`: Implements transfer learning attempts using pre-trained CNN models like AlexNet and VGG16 to extract features and test their performance on the dementia classification task.
- `attempt/ANNtest.py`: This script is used for training the custom ANN model on the numerical data extracted from MRI images, optimizing it for dementia classification.
- `attempt/CNNcustomModel.py`: This script handles the training of the custom CNN model on the MRI image data, incorporating advanced training strategies to enhance performance.
- `attempt/FUSION.py`: Responsible for implementing the training of the Fusion model, combining outputs from both the CNN and ANN models to achieve superior classification results.
- `pretrained_model_script/*`: This folder contains Python scripts developed for initial experiments with pre-trained models, including earlier attempts at transfer learning and feature extraction. These scripts were used to evaluate models like VGG16, AlexNet, Unet, ResNet34, and DenseNet121 before shifting focus to custom models and the fusion approach.

The scripts in the `pretrained_model_script` folder represent the foundation of our initial explorations into using existing deep learning models, providing valuable insights that informed the design of the custom CNN, ANN, and Fusion models used in the final stages of the project.



### Outputs
- Model performance metrics will be displayed in the notebook.
- The final model weights can be saved and used for further analysis or real-world applications.

---

Feel free to contribute to this project by exploring new fusion strategies, expanding the dataset, or integrating the models into clinical applications. For any questions or feedback, please open an issue or contact us at [email@example.com].
