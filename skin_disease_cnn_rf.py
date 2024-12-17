# -*- coding: utf-8 -*-
"""
Skin Disease Detection using CNN and Random Forest
Refactored and Optimized
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical

# ============================
# 1. DATA LOADING FUNCTION
# ============================
def load_data(data_dir, image_size=128):
    images, labels = [], []
    for folder in glob.glob(os.path.join(data_dir, "*")):
        label = os.path.basename(folder)
        for img_path in glob.glob(os.path.join(folder, "*.jpg")):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (image_size, image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensuring consistent color format
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# ============================
# 2. DATA PREPROCESSING FUNCTION
# ============================
def preprocess_data(x, y):
    x = x / 255.0  # Normalize pixel values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_one_hot = to_categorical(y_encoded)
    return x, y_encoded, y_one_hot, le

# ============================
# 3. CNN MODEL FUNCTION
# ============================
def build_cnn_model(input_shape, activation_fn='relu'):
    model = Sequential([
        Conv2D(32, kernel_size=3, activation=activation_fn, padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, kernel_size=3, activation=activation_fn, padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, kernel_size=3, activation=activation_fn, padding='same'),
        BatchNormalization(),
        Conv2D(64, kernel_size=3, activation=activation_fn, padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation=activation_fn),
        Dense(4, activation='softmax')  # 4 Classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ============================
# 4. CONFUSION MATRIX FUNCTION
# ============================
def plot_confusion_matrix(true_labels, predicted_labels, label_encoder):
    cm = confusion_matrix(true_labels, predicted_labels)
    sns.heatmap(cm, annot=True, cmap='coolwarm', fmt='d',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.show()

# ============================
# MAIN FUNCTION
# ============================
def main():
    # Load Training and Test Data
    image_size = 128
    train_dir = "Skin Images/new"
    test_dir = "Skin Images/new test"

    print("Loading Training Data...")
    x_train, y_train = load_data(train_dir, image_size)
    print("Loading Testing Data...")
    x_test, y_test = load_data(test_dir, image_size)

    # Preprocess Data
    x_train, y_train_enc, y_train_one_hot, le = preprocess_data(x_train, y_train)
    x_test, y_test_enc, y_test_one_hot, _ = preprocess_data(x_test, y_test)

    # Build and Train CNN
    print("Building CNN Model...")
    cnn_model = build_cnn_model(input_shape=(image_size, image_size, 3))
    print(cnn_model.summary())

    print("Training CNN...")
    history = cnn_model.fit(x_train, y_train_one_hot, epochs=30, validation_data=(x_test, y_test_one_hot))

    # Plot Accuracy and Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.show()

    # Evaluate on Test Data
    predictions_cnn = np.argmax(cnn_model.predict(x_test), axis=-1)
    predictions_cnn_labels = le.inverse_transform(predictions_cnn)
    print("\n--- CNN Confusion Matrix ---")
    plot_confusion_matrix(y_test, predictions_cnn_labels, le)

    # Feature Extraction for Random Forest
    feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)
    X_train_features = feature_extractor.predict(x_train)
    X_test_features = feature_extractor.predict(x_test)

    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_features, y_train_enc)

    # Evaluate Random Forest
    rf_predictions = rf_model.predict(X_test_features)
    rf_predictions_labels = le.inverse_transform(rf_predictions)
    print("\n--- Random Forest Accuracy ---")
    print("Accuracy:", accuracy_score(y_test, rf_predictions_labels))
    print("\n--- Random Forest Confusion Matrix ---")
    plot_confusion_matrix(y_test, rf_predictions_labels, le)

    # Test Single Image
    test_idx = 5
    img = x_test[test_idx]
    plt.imshow(img)
    plt.show()

    input_img = np.expand_dims(img, axis=0)
    features = feature_extractor.predict(input_img)
    prediction_rf = rf_model.predict(features)
    prediction_label = le.inverse_transform(prediction_rf)

    print(f"Random Forest Prediction: {prediction_label[0]}")
    print(f"Actual Label: {y_test[test_idx]}")

if __name__ == "__main__":
    main()
