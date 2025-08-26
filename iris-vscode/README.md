Iris Flower Species Classification
Project Description
This project demonstrates a fundamental machine learning task: classifying Iris flowers into their correct species based on their physical measurements. Using the well-known Iris dataset, a k-Nearest Neighbors (k-NN) model is trained and evaluated to accurately distinguish between the three species: Iris-setosa, Iris-versicolor, and Iris-virginica.

Objective
The primary objective of this project is to build a machine learning model that learns from the sepal and petal measurements of Iris flowers and predicts their species with high accuracy.

Technologies Used
Python: The core programming language for the project.

pandas: Used for loading and manipulating the dataset.

scikit-learn: The primary machine learning library used for implementing the k-NN model, splitting the data, and calculating performance metrics.

How to Run the Project
Follow these steps to set up and run the classification model:

Step 1: Prerequisites
Ensure you have Python installed on your system.

Step 2: Install Dependencies
Open your terminal or command prompt and install the required libraries using pip:

pip install pandas scikit-learn

Step 3: File Setup
Create a project folder (e.g., iris-classification).

Place the IRIS.csv dataset file inside this folder.

Create a Python script named classify_iris.py in the same folder and add the provided code to it.

Step 4: Execute the Script
Run the Python script from your terminal:

python classify_iris.py

Dataset
The project uses the classic Iris flower dataset, which contains:

150 samples of Iris flowers.

4 features: sepal_length, sepal_width, petal_length, and petal_width (all in cm).

1 target variable: species, with three possible values: Iris-setosa, Iris-versicolor, and Iris-virginica.

Methodology
Data Loading: The IRIS.csv file is loaded into a pandas DataFrame.

Data Preprocessing: The categorical species column is converted into a numerical format using LabelEncoder.

Data Splitting: The dataset is divided into a training set (80%) and a testing set (20%) to ensure the model is evaluated on unseen data.

Model Training: A KNeighborsClassifier model is trained on the training data.

Model Evaluation: The model's predictions are compared against the actual species in the test set to calculate accuracy and generate a detailed classification report and confusion matrix.

Results
The model achieved an accuracy of 100% on the test data. This indicates that the k-NN classifier was able to perfectly distinguish between the different Iris species using the given measurements. The classification report and confusion matrix provide further details on the model's perfect performance for each class.
