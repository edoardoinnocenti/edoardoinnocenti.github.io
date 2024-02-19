---
title: "Digit Recognition"
excerpt: "Implementation of a Deep Learning model (Convolutional Neural Network) for the recognition of hand-written digits <br><br><img src='/images/DigitRecognizer_38_0.png'>"
collection: portfolio
---

## Introduction
This project focuses on developing a machine learning model to identify handwritten digits (0-9) using a reduced version of the MNIST dataset. It's a cornerstone in understanding image recognition techniques in the field of machine learning.
The proposed model shows very good performance with a final accuracy on the unknown dataset **higher than 0.98%**.

Full project github repository: [Click Here!](https://github.com/edoardoinnocenti/DigitRecognizer/blob/main/DigitRecognizer.ipynb)

<div style="display: flex; justify-content: space-between;">
    <img src="/images/DigitRecognizer_11_1.png" alt="alt text">
    <img src="/images/DigitRecognizer_38_0.png" alt="alt text">
</div>
<br>

## Dataset Overview
The reduced version of the MNIST dataset, containing thousands of labeled images of handwritten digits, is utilized. Each image is a grayscale 28x28 pixel representation of a digit.

## Preprocessing Steps
- **Data Loading**: Loading the dataset into training and test sets.
- **Normalization**: Normalizing pixel values to a range of 0 to 1.
- **Reshaping**: Adjusting images to fit the model's input requirements.

## Model Development
- **Architecture**: Define the Architercure of the Deep Learning model.
- **Compilation**: Compilation with chosen optimizer and loss function.
- **Training**: Model training over several epochs.

## Evaluation and Testing
- **Validation**: Performance validation is made on a subset validation set.
- **Testing**: Testing against an unseen test dataset is directly made with the submission to Kaggle.

## Model Architecture:
The used model is a Convolutional Neural Network (CNN) designed for image classification tasks. It follows a typical CNN architecture pattern, consisting of an input layer, several hidden layers, and a fully connected layer for classification. Here's a breakdown of its structure:

**Input Layer**: It starts with a 2D convolutional layer with 32 filters of size 3x3 and ReLU activation, designed to process images of size 28x28 with a single color channel (e.g., grayscale images). This is followed by a max pooling layer that reduces the spatial dimensions by half, using a 2x2 pool size and a stride of 2.

**Hidden Layers**: It includes two more sets of convolutional and max pooling layers. The second convolutional layer has 64 filters, uses the same kernel size, and applies padding to maintain the spatial dimensions of the output. Each of these layers is followed by max pooling, further reducing the dimensions.

**Flatten Layer**: After the convolutional and pooling layers, the model flattens the 2D feature maps into a 1D vector, making it possible to connect to dense layers for classification.

**Fully Connected Layers**: The flattened output is then passed through two dense layers with 64 and 128 neurons respectively, each followed by a dropout layer with a dropout rate of 0.2 to prevent overfitting. These layers use ReLU activation.

**Output Layer**: The final dense layer has 10 neurons with softmax activation, making it suitable for classification tasks with 10 classes. The softmax activation ensures the output values are in the range 0-1 and sum up to 1, acting as probabilities for each class.

Overall, this CNN model is configured to extract features from images through convolutional and pooling layers, reduce overfitting with dropout, and classify images into one of 10 categories using a softmax output layer.

## Results
The training results are shown in the code, where model training, confusion matrix and correct/incorrect images are reported.
The final sumbission.csv returns an accuracy > 0.98 % on the unknown test dataset.