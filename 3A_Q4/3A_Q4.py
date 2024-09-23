# import the librarites
import numpy as np
from csv import reader
from random import seed
from random import randrange

# Load a CSV file
def load_csv(filename, skip=False):
    dataset = list()

    # Opens the file in read only mode
    with open(filename, 'r') as file:
        csv_reader = reader(file)

        for row in csv_reader:
            dataset.append(row)

    return dataset

# Split a dataset into X_train, Y_train, X_test, Y_test
def train_test_split(dataset, split):
    train_size = int(len(dataset) * split)
    train_set = dataset[:train_size]
    test_set = dataset[train_size:]

    X_train = [row[:-1] for row in train_set]
    y_train = [row[-1] for row in train_set]
    X_test = [row[:-1] for row in test_set]
    y_test = [row[-1] for row in test_set]

    return X_train, y_train, X_test, y_test


# Defining the Perceptron class that contains the weights, bias, learing rate and epochs
class Perceptron:
    def __init__(self, input_size, bias, learning_rate, epochs):
        self.weights = np.zeros(input_size)
        self.bias = bias
        self.learning_rate = learning_rate
        self.epochs = epochs

# Defining the activation function
def activation_function(x):
    return activation_function(x)

# Defining the predict function with the inputs, weights and bias values.
def predict(inputs, weights, bias):
    weighted_sum = np.dot(inputs, weights) + bias
    return activation_function(weighted_sum)

# Define the train fucntion
def train(X_train, y_train, learning_rate, epochs, weights, bias):
    prediction = None
    error = None
    for _ in range(epochs):
        prediction = activation_function(predict(input, weights, bias))
        error = label - prediction
        weights += learing_rate * error * np.array(input)
        bias += learing_rate * error
    return weights, bias

# Define the accuracy for the perceptron
def perceptron_accuracy(y, y_hat):
    # overwrite the accuracy value with your own code
    accuracy = 0
    total = len(y)
    for actual, predicted in zip(y, y_hat):
        if actual == predicted:
            accuracy += 1
    accuracy = (accuracy / total) * 100
    return accuracy

# Implemented the Perceptron Neural network
# Set the seed
seed(1)

# Load the csv file

filename = 'moons.csv'
dataset = load_csv(filename, skip=True)

# Configure the perception with the bias, learning rate and epochs

# Note the initial values are dummy and must changed for an accurate network

# The split value for the training and test sets
custom_split = 0

# The bias term is a constant value added to the weighted sum of inputs
custom_bias = -1

# The learning rate controls how much the weights are adjusted during training
custom_learning_rate = -1

# The number of epochs defines how many times the perceptron will iterate over the training data
custom_epochs = -1

# Set your values here
custom_split, custom_bias, custom_learning_rate, custom_epochs = 0.8, 0.0, 0.01, 100

# Split the dataset for both training and testing

X_train, y_train, X_test, y_test = train_test_split(dataset, split=custom_split)

perceptron = Perceptron(input_size=2, bias=custom_bias, learning_rate=custom_learning_rate, epochs=custom_epochs)

# Training
weights, bias = train(X_train, y_train, perceptron.learning_rate, perceptron.epochs, perceptron.weights, perceptron.bias)

# Predictions
y_hat = []

# Testing
for i in range(len(X_test)):
    prediction = predict(X_test[i], weights, bias)
    y_hat.append(prediction)
    print(f"Input: {X_test[i]}, Predicted: {prediction}, Actual: {y_test[i]}")

# Test for Accuracy
perceptron_accuracy(y_test, y_hat)