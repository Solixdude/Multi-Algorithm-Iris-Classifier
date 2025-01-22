Iris Classification Project
Overview

This project demonstrates the classification of Iris flower species using machine learning models. Multiple algorithms are compared for accuracy, and the best-performing model is saved for future use. The dataset used is the well-known Iris Dataset, which contains information about the sepal and petal dimensions of three Iris species: setosa, versicolor, and virginica.
Features

    Load and preprocess the Iris dataset.
    Split data into training and validation sets.
    Compare multiple classification algorithms:
        Logistic Regression
        Linear Discriminant Analysis
        K-Nearest Neighbors
        Decision Tree
        Gaussian Naive Bayes
        Support Vector Machine
    Evaluate models using cross-validation.
    Visualize performance using boxplots.
    Train the best model on the training set.
    Evaluate the model on the validation set.
    Save and load the trained model for reuse.

Requirements

To run the project, you need the following Python libraries:

    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    joblib

Files

    iris_classifier.py: Main script that performs data loading, model training, evaluation, and saving/loading the best model.
    iris_best_model.pkl: Saved model file for the best-performing algorithm.

License

This project is licensed under the MIT License.

