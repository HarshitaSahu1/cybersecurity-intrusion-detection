# 🛡️ Intrusion Detection System using Machine Learning

## 📌 Project Overview

This project presents a **Machine Learning-based Intrusion Detection System (IDS)** capable of identifying malicious network traffic by performing both **Binary Classification** (Normal vs Attack) and **Multiclass Classification** (different attack categories).

The objective is to build a reliable intrusion detection model by handling class imbalance, selecting the most relevant features, and evaluating the model using multiple classification metrics.

---

## 🎯 Objectives

* Detect whether network traffic is **Normal** or **Attack**.
* Classify different categories of network attacks.
* Handle severe class imbalance using **SMOTE**.
* Select the most informative features using **Recursive Feature Elimination (RFE)**.
* Train and evaluate a **Random Forest Classifier**.
* Compare Binary and Multiclass classification performance.

---

## 📊 Dataset

The project uses a network intrusion detection dataset containing various network connection features such as:

* Duration
* Protocol Type
* Service
* Flag
* Source Bytes
* Destination Bytes
* Login Information
* Traffic Statistics
* Error Rates
* Host Features
* Network Behavior Features

The target variable represents different network attack categories along with normal network traffic.

---

## 🧹 Data Preprocessing

The following preprocessing techniques were applied:

* Handling missing values (if present)
* Removing unnecessary columns
* Label Encoding of categorical variables
* Feature scaling (where required)
* Train-Test Split
* Handling class imbalance using **SMOTE**

---

## ⚖️ Handling Class Imbalance

The dataset contained highly imbalanced attack classes.

To improve model learning:

* SMOTE (Synthetic Minority Oversampling Technique) was applied only on the training dataset.
* Extremely low-frequency attack categories were combined into **Other_Attacks** to improve multiclass classification stability.

Merged classes:

* bufferoverflow
* ftpwrite
* guesspassword
* rootkit

---

## 🎯 Feature Selection

Feature selection was performed using **Recursive Feature Elimination (RFE)** with a **Random Forest Classifier**.

Benefits:

* Reduced dimensionality
* Faster model training
* Removal of less informative features
* Improved model interpretability

---

## 🤖 Machine Learning Model

**Algorithm Used**

* Random Forest Classifier

Reasons for choosing Random Forest:

* Handles high-dimensional data efficiently
* Robust against overfitting
* Provides feature importance
* Performs well on classification problems
* Suitable for imbalanced datasets after SMOTE

---

# Binary Classification

### Classes

* Normal
* Attack

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix

---

# Multiclass Classification

### Classes

* Normal
* Neptune
* Smurf
* Satan
* Portsweep
* Back
* Other_Attacks

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix

---

## 📈 Visualizations

The project includes visualizations such as:

* Class Distribution
* Correlation Heatmap
* Feature Importance
* ROC Curve
* Confusion Matrix

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)

---

## 🚀 Key Machine Learning Techniques

* Data Cleaning
* Label Encoding
* Train-Test Split
* SMOTE
* Recursive Feature Elimination (RFE)
* Random Forest Classification
* Binary Classification
* Multiclass Classification
* Model Evaluation

---

## 📊 Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC Curve
* ROC-AUC Score
* Confusion Matrix
* Classification Report

These metrics provide a comprehensive understanding of model performance beyond overall accuracy.

---

## 🌐 Live Demo

The Intrusion Detection System has been deployed using **Streamlit Cloud** and can be accessed here:

🔗 **Live Application:** https://cybersecurity-intrusion-detection-4vtn63pubargrpe8xtnwuq.streamlit.app/

### Demo Login Credentials

| Username | Password |
|----------|----------|
| admin | 123 |

> **Note:** This is a demonstration application intended for showcasing the machine learning model and dashboard. The login credentials are provided for evaluation purposes only.

## 💡 Future Improvements

Deploy the application on cloud platforms such as AWS, Azure, or GCP.
Implement secure user authentication and role-based access control.
Integrate real-time network traffic monitoring.
Evaluate additional machine learning algorithms such as XGBoost, LightGBM, and Support Vector Machines.
Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
Implement deep learning-based intrusion detection models

---

## 👩‍💻 Author

**Harshita Sahu**

Aspiring Data Analyst | Machine Learning Enthusiast

### Skills

* Python
* SQL
* Machine Learning
* Data Analysis
* Feature Engineering
* Model Evaluation
* Data Visualization

---

## ⭐ Project Highlights

* Built Binary and Multiclass Intrusion Detection Models
* Addressed severe class imbalance using SMOTE
* Applied Recursive Feature Elimination (RFE)
* Trained Random Forest Classifier
* Evaluated models using multiple performance metrics
* Performed feature engineering and preprocessing for improved model performance

