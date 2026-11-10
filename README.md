# 🎗️ Breast Cancer Prediction Classification Using Artificial Neural Network (ANN)

##  Project Overview

This project focuses on developing a Deep Learning classification model using an **Artificial Neural Network (ANN)** to predict whether a breast tumor is **Benign (Non-Cancerous) or Malignant (Cancerous)** based on various tumor characteristics.

The model learns patterns from numerical features such as radius, texture, perimeter, area, smoothness, compactness, concavity, and other tumor measurements to classify the tumor into one of two categories.

The primary objective is to explore how Artificial Neural Networks can be applied to binary classification problems using Python, TensorFlow, and Keras.

##  Business / Problem Objective

Breast cancer classification involves analyzing different characteristics of tumors to distinguish between benign and malignant cases.

The objective of this project is to:

- Analyze breast cancer data through Exploratory Data Analysis (EDA).
- Understand relationships between different tumor characteristics.
- Preprocess and standardize numerical features.
- Build an Artificial Neural Network classification model.
- Evaluate model performance using test data.
- Develop a model that predicts tumor classification based on input features.

##  Dataset Information

**Dataset:** Breast Cancer Wisconsin Dataset

**Source:** Scikit-learn Breast Cancer Dataset

Dataset characteristics:

- Total Records: 56569
- Total Input Features: 30
- Target Variable: Diagnosis / Label
- Problem Type: Classification
- Number of Classes: 2

### Target Classes

| Label | Diagnosis | Meaning |
|---|---|---|
| 0 | Malignant | Cancerous |
| 1 | Benign | Non-Cancerous |

Note: The ANN training section uses the Scikit-learn dataset's original target encoding. The separate CSV EDA section uses LabelEncoder on the diagnosis column, where Benign = 0 and Malignant = 1.

### Important Features

The dataset contains 30 numerical features grouped into three categories:

**1. Mean Features**
- Radius Mean
- Texture Mean
- Perimeter Mean
- Area Mean
- Smoothness Mean
- Compactness Mean
- Concavity Mean
- Concave Points Mean
- Symmetry Mean
- Fractal Dimension Mean

**2. Standard Error Features**
- Radius SE
- Texture SE
- Perimeter SE
- Area SE
- Smoothness SE
- Compactness SE
- Concavity SE
- Concave Points SE
- Symmetry SE
- Fractal Dimension SE

**3. Worst Features**
- Radius Worst
- Texture Worst
- Perimeter Worst
- Area Worst
- Smoothness Worst
- Compactness Worst
- Concavity Worst
- Concave Points Worst
- Symmetry Worst
- Fractal Dimension Worst

##  Technologies and Tools Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras
- Jupyter Notebook
- Streamlit

##  Project Workflow

### 1. Data Collection
Loaded the Breast Cancer Wisconsin dataset using Scikit-learn and explored the separate CSV dataset during the initial EDA process.

### 2. Data Cleaning
- Removed unnecessary columns such as ID and Unnamed columns.
- Checked missing values.
- Checked duplicate records.
- Inspected numerical feature distributions.

### 3. Exploratory Data Analysis (EDA)
- Analyzed the distribution of Benign and Malignant cases.
- Explored numerical feature distributions.
- Calculated descriptive statistics.
- Examined skewness.
- Analyzed correlations between numerical features.
- Visualized relationships using Matplotlib and Seaborn.

### 4. Data Preprocessing
- Separated independent and dependent variables.
- Split the dataset into training and testing sets.
- Applied StandardScaler to standardize numerical features.
- Used fit_transform() on training data and transform() on testing data.

### 5. Model Building – Artificial Neural Network

Developed an ANN architecture using TensorFlow and Keras.

**ANN Architecture:**

| Layer | Configuration | Activation |
|---|---|---|
| Input Layer | 30 input features | — |
| Flatten Layer | Input shape (30,) | — |
| Hidden Layer | 20 neurons | ReLU |
| Output Layer | 2 neurons | Sigmoid |

**Model Configuration:**
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Evaluation Metric: Accuracy
- Epochs: 10
- Validation Split: 10%

### 6. Model Training
Trained the ANN model using standardized training data and monitored training and validation accuracy and loss across 10 epochs.

### 7. Model Evaluation

Evaluated the trained ANN model on unseen test data.

**Test Accuracy: 97.37%**

The model achieved approximately 97.37% accuracy on the test dataset.

## Key Highlights

- Worked with 569 breast cancer records and 30 numerical features.
- Performed data cleaning and exploratory data analysis.
- Applied StandardScaler for feature standardization.
- Built an Artificial Neural Network using TensorFlow and Keras.
- Implemented ReLU and Sigmoid activation functions.
- Trained the neural network using the Adam optimizer.
- Evaluated the model using test data.
- Achieved 97.37% test accuracy.
- Explored a Streamlit interface for interactive predictions.

## Streamlit Application

Developed a user-friendly web application using Streamlit to demonstrate the trained ANN model.

The application allows users to:
- Enter tumor measurement features.
- Submit input values to the trained ANN model.
- Obtain a predicted classification.
- View the classification result as Benign or Malignant.

## Skills Gained

Through this project, I strengthened my practical knowledge of:

- Python Programming
- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Data Visualization
- Feature Standardization
- Deep Learning
- Artificial Neural Networks (ANN)
- TensorFlow and Keras
- Classification Algorithms
- Model Training and Evaluation
- Streamlit Application Development


##  Disclaimer

This project is developed for educational and learning purposes only. It is not a clinically validated medical diagnostic system. Predictions should not be used as a substitute for professional medical advice, diagnosis, or treatment.

##  Author

**Yasala Akshitha**

Aspiring Data Scientist | Machine Learning & Deep Learning Enthusiast

LinkedIn: https://linkedin.com/in/akshithayasala

---
