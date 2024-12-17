# Skin Disease Detection using CNN and Random Forest

## Overview
This project implements a **Skin Disease Detection System** using **Convolutional Neural Networks (CNN)** for feature extraction . The system processes skin images, extracts features using a CNN, and combines them with Random Forest to achieve robust accuracy for skin disease detection.

## Features
- **Image Preprocessing**: Resizes and normalizes images for consistent input.
- **CNN-based Feature Extraction**: Extracts visual features using a CNN model.
- **Classification**: Combines CNN and Random Forest for improved performance.
  
---

## Dataset
The system uses labeled images of different skin diseases stored in different sites.



The images are classified into 4 classes:
1. Class A
2. Class B
3. Class C
4. Class D

> Ensure the dataset is organized as follows:
```
Skin Images/
├── new/          # Training Data
│   ├── Class_A/  # Images for Class A
│   ├── Class_B/  # Images for Class B
│   ├── ...
├── new test/     # Testing Data
│   ├── Class_A/
│   ├── Class_B/
│   ├── ...
```

---

## Prerequisites
Ensure you have the following libraries installed:

- Python 3.x
- OpenCV
- TensorFlow/Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Install required libraries:
```bash
pip install numpy opencv-python matplotlib seaborn scikit-learn tensorflow keras
```

---

## How to Run
1. **Setup Dataset**: Place your training and testing images in the respective directories as shown above.
2. **Run the Script**:
   - Save the provided Python file as `skin_disease_cnn_rf.py`.
   - Execute the script:
     ```bash
     python skin_disease_cnn_rf.py
     ```
3. **Output**:
   - Training and Validation Accuracy and Loss Graphs.
   - Confusion Matrices for CNN and Random Forest.
   - Accuracy Scores for CNN and Random Forest.
   - Visual Prediction Results on selected test images.

---

## Workflow
1. **Data Loading**:
   - Training and test images are resized to `128x128` and normalized.
2. **Model Training**:
   - A CNN is trained on the training dataset.
   - Features are extracted using the CNN for input to Random Forest.
3. **Classification**:
   - CNN performs initial classification.
   - Random Forest is trained on the extracted CNN features for enhanced classification.
4. **Evaluation**:
   - Both models are evaluated using accuracy and confusion matrices.
5. **Visualization**:
   - Plots accuracy, loss curves, and confusion matrices.
   - Displays predictions on sample test images.

---

## Output Examples
### Accuracy and Loss Graphs
- Training and validation curves help analyze model performance.

### Confusion Matrix
- CNN Confusion Matrix
- Random Forest Confusion Matrix

### Predictions
Displays test image predictions and actual labels:
```
Random Forest Prediction: Class_B
Actual Label: Class_B
```


