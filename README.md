# Fer-ml-model
Facial Expression Recognition (FER) Model
Overview
This repository contains code for training and deploying a Facial Expression Recognition (FER) model using Convolutional Neural Networks (CNNs). The model is trained on the FER2013 dataset, which consists of images of facial expressions categorized into seven classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.

Model Architecture
The model architecture used is a CNN consisting of multiple convolutional layers followed by max-pooling layers and fully connected layers. The final layer utilizes softmax activation for multi-class classification.

Input Image Shape: (48, 48, 1)
Output Classes: 7 (One for each facial expression category)
Dataset
The FER2013 dataset is used for training and testing the model. It contains grayscale images of facial expressions with corresponding labels.

Training Data Directory: /kaggle/input/fer2013/train
Validation Data Directory: /kaggle/input/fer2013/test
Image Size: 48x48 pixels
Batch Size: 32
Number of Epochs: 25
Training
The model is trained using an ImageDataGenerator for data augmentation and flow_from_directory to load images directly from directories. Training is performed using the Adam optimizer and categorical cross-entropy loss function.

Evaluation
The model's performance is evaluated on a separate validation set using accuracy as the metric.

Training Accuracy: 52% (After 25 epochs)
Model File
The trained model is saved as emotion_detection_model.h5.

Sample Predictions
A sample of test images with their predicted labels is displayed to demonstrate the model's performance.

Usage
Clone the repository.
Install the required dependencies (keras, matplotlib, numpy).
Run the notebook to train and evaluate the model.
Use the trained model for facial expression recognition tasks.
Note
This model is trained on the FER2013 dataset and may not generalize well to other datasets or real-world scenarios.
Further fine-tuning or transfer learning may be necessary for improved performance.
Link of the dataset -https://www.kaggle.com/datasets/msambare/fer2013
