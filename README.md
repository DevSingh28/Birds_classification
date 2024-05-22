# Bird Species Classification Model

This repository contains a convolutional neural network (CNN) model for classifying images of bird species. The model is trained on a dataset of 100 different bird species and is capable of predicting the species of a bird from an image with high accuracy.

## Project Overview

The goal of this project is to build a deep learning model that can identify different species of birds from images. The project uses TensorFlow and Keras for model building and training, and includes steps for data preprocessing, data augmentation, and model evaluation.

## Technologies Used

- **TensorFlow and Keras**: For building and training the deep learning model.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization.
- **Google Colab**: For leveraging GPU resources to train the model.

## Dataset

The dataset used for this project can be downloaded from Kaggle: [100 Bird Species](https://www.kaggle.com/datasets/gpiosenka/100-bird-species). It contains images of 100 different bird species, divided into training, validation, and test sets.

## Key Steps in the Project

1. **Environment Setup**: Set up the necessary environment, including downloading the dataset from Kaggle and setting up directories for training, validation, and test data.
2. **Data Loading and Preprocessing**: Load and preprocess the images, including resizing and normalizing.
3. **Data Augmentation**: Apply data augmentation techniques to improve model generalization.
4. **Model Building**: Create a CNN model with multiple convolutional and pooling layers.
5. **Model Compilation and Training**: Compile the model with appropriate loss function and optimizer, and train it using the training dataset.
6. **Model Evaluation**: Evaluate the model on the test dataset to assess its performance.
7. **Prediction**: Implement a function to make predictions on new images.
8. **Model Saving**: Save the trained model for future use.

## Instructions

1. **Setup and Install Dependencies**:
   - Ensure you have Python 3.x installed.
   - Install the required libraries using pip:
     ```sh
     pip install tensorflow pandas matplotlib
     ```

2. **Download and Extract Dataset**:
   - Download the dataset from Kaggle and extract it:
     ```python
     ! mkdir ~/.kaggle
     ! cp kaggle.json ~/.kaggle/
     ! chmod 600 ~/.kaggle/kaggle.json
     ! kaggle datasets download gpiosenka/100-bird-species
     ! unzip 100-bird-species.zip
     ```

3. **Load Data**:
   - Set up the paths to the data directories and load the data using `tf.keras.preprocessing.image_dataset_from_directory`:
     ```python
     train_dir = '/content/train'
     test_dir = '/content/test'
     valid_dir = '/content/valid'

     train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir, batch_size=32, image_size=(224, 224))
     test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir, batch_size=32, image_size=(224, 224))
     valid_data = tf.keras.preprocessing.image_dataset_from_directory(valid_dir, batch_size=32, image_size=(224, 224))
     ```

4. **Data Augmentation**:
   - Apply data augmentation to the training data:
     ```python
     from tensorflow.keras.layers.experimental import preprocessing
     data_aug_layer = tf.keras.Sequential([
         preprocessing.RandomFlip('horizontal_and_vertical'),
         preprocessing.RandomRotation(0.2),
         preprocessing.RandomZoom(0.2),
         preprocessing.RandomHeight(0.2),
         preprocessing.RandomWidth(0.2),
     ])
     ```

5. **Build and Compile Model**:
   - Define the model architecture and compile it:
     ```python
     from tensorflow.keras import layers, models
     inputs = tf.keras.Input(shape=(224, 224, 3))
     x = data_aug_layer(inputs)
     x = layers.Rescaling(1./255)(inputs)
     x = layers.Conv2D(32, 3, activation='relu')(x)
     x = layers.MaxPooling2D()(x)
     x = layers.Conv2D(64, 3, activation='relu')(x)
     x = layers.MaxPooling2D()(x)
     x = layers.Conv2D(128, 3, activation='relu')(x)
     x = layers.GlobalAveragePooling2D()(x)
     x = layers.Dropout(0.5)(x)
     outputs = layers.Dense(len(train_data.class_names), activation='softmax')(x)
     model = models.Model(inputs, outputs)
     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
     ```

6. **Train the Model**:
   - Train the model using the training and validation data:
     ```python
     history = model.fit(train_data, epochs=20, validation_data=valid_data)
     ```

7. **Evaluate the Model**:
   - Evaluate the model on the test dataset:
     ```python
     model.evaluate(test_data)
     ```

8. **Make Predictions**:
   - Implement a function to predict bird species from new images:
     ```python
     from tensorflow.keras.utils import load_img, img_to_array
     def predictor(filename, class_names):
         img = load_img(filename, target_size=(224, 224))
         img_array = img_to_array(img)
         img_array = tf.expand_dims(img_array, 0)
         predictions = model.predict(img_array)
         predicted_class = class_names[tf.argmax(predictions[0])]
         confidence = tf.reduce_max(predictions[0])
         plt.imshow(img)
         plt.title(f"Prediction: {predicted_class}, Confidence: {confidence:.2f}")
         plt.axis('off')
         plt.show()
     ```

9. **Save the Model**:
   - Save the trained model for future use:
     ```python
     model.save('my_model')
     model.save(filepath='/content/drive/MyDrive')
     ```


## Acknowledgements

- The dataset was provided by [Kaggle](https://www.kaggle.com/).
- This project was developed using TensorFlow and Keras.
