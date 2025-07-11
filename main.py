import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds




datagen = ImageDataGenerator(
    rescale = 1/255,  
    validation_split = .2, 
)

train_data = datagen.flow_from_directory(
    'asl_alphabet/asl_dataset', 
    target_size= (224, 224), 
    batch_size=32,    
    class_mode='categorical', 
    subset='training'    
)

test_data = datagen.flow_from_directory(
    'asl_alphabet/asl_dataset', 
    target_size= (224, 224), 
    batch_size=32,               
    class_mode='categorical', 
    subset='validation'    
)

train_data = datagen.flow_from_directory(
    'asl_alphabet/asl_dataset', 
    target_size= (224, 224),
    batch_size=32,            
    class_mode='categorical', 
    subset='training'    
)

test_data = datagen.flow_from_directory(
    'asl_alphabet/asl_dataset', 
    target_size= (224, 224), 
    batch_size=32,
    class_mode='categorical', 
    subset='validation'    
)

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)), 
    layers.Conv2D(32, (3,3), activation='relu'), 
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'), 
    layers.MaxPooling2D(2,2),                    
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2), 
    layers.Flatten(), 
    layers.Dense(128, activation='relu'), 
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='tanh'),  
    layers.Dense(36, activation='softmax') 
])
model.compile( 
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy'] 
)

history = model.fit(train_data, epochs=10, validation_data=test_data)
history.history['accuracy']
history.history['val_accuracy']
plt.plot(history.history['val_accuracy'],label="Validation Accuracy", color='blue')
plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Time')
plt.legend()
plt.show()

model.save("ASL_model1.0.keras")
