# ASL Hand Sign Recognition Neural Network

This project is a Convolutional Neural Network (CNN) built using TensorFlow/Keras for recognizing **American Sign Language (ASL)** hand signs from images.

It trains on 36 classes representing ASL alphabet letters (A–Z) and some additional symbols.



## Features

- Image classification using deep CNN
- Real-time training and validation accuracy plotting
- Input shape: `224x224x3` RGB images
- 36-class multi-class classification using `softmax`



## Model Architecture

```text
Input: (224, 224, 3)
↓
Conv2D(32) → MaxPooling2D
↓
Conv2D(64) → MaxPooling2D
↓
Conv2D(128) → MaxPooling2D
↓
Flatten → Dense(128) → Dense(64) → Dense(32)
↓
Output: Dense(36, activation='softmax')
