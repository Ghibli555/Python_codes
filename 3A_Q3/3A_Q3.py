# import the libraries
import numpy as np
from csv import reader
# needed for displaying the classification graph.
import matplotlib.pyplot as plt

# load the csv file.
def load_csv(filename, skip = False):
    dataset = list()
    # Opens the file in read only mode
    with open(filename, 'r') as file:
        csv_reader = reader(file)

        # Skip the header row
        if skip:
            next(csv_reader, None)

        for row in csv_reader:
            dataset.append(row)
    return dataset

# Encode mailgnant values, M to 0 and begin, B values to 1
def diagnosis_column_to_number(dataset, column):
    # Convert 'M' to 0 and 'B' to 1
    for row in dataset:
        if (row[column] == 'M'):
            row[column] = 0
        elif (row[column] == 'B'):
            row[column] = 1

    return dataset

# Extract only the x data
def extract_only_x_data(dataset):
    if len(dataset) == 0:
        return

    data = list()

    for i in range(0, len(dataset)):
        data.append(list())

        for j in range(0, len(dataset[i]) - 1):
            data[-1].append(float(dataset[i][j]))

    return data

# Extract only the y data
def extract_only_y_data(dataset):
    if len(dataset) == 0:
        return

    data = list()

    for i in range(0, len(dataset)):
        data.append(int(dataset[i][-1]))

    return data

# Define the sigmoid function
def sigmoid(z):
#calculate sigmoid of z (the formula s(z)=1+e−z1)
 z = 1 / (1 + np.exp(-z))
# Return the value of the implemented sigmoid function, do not simply return z
 return z

# Define the loss function
def loss(y_hat):
    # overwrite the loss value with your own code
    loss = 0
    m = y.shape[0]
    loss = -1/m * (np.dot(y, np.log(y_hat)) + np.dot((1 - y), np.log(1 - y_hat)))

    # Return the value of the implemented loss function, do not simply return loss of zero
    return loss

# Define the gradients fucntion
def gradients(X, y, y_hat):
    # number of training examples.
    number_of_examples = X.shape[0]

    # Gradient of loss weights.
    dw = (1 / number_of_examples) * np.dot(X.T, (y_hat - y))

    # Gradient of loss bias.
    db = (1 / number_of_examples) * np.sum((y_hat - y))

    return dw, db

# Train the dataset
def train(X, y, batch_size, epochs, learning_rate):
    number_of_examples, number_of_features = X.shape

    print(number_of_examples)
    print(number_of_features)

    # Initializing weights and bias to zeros.
    weights = np.zeros((number_of_features, 1))
    bias = 0

    # Reshaping y.
    y = y.reshape(number_of_examples, 1)

    # Empty list to store losses.
    losses = []

    # Training loop.
    for epoch in range(epochs):
        for i in range((number_of_examples - 1) // batch_size + 1):
            # Defining batches. SGD.
            start_i = i * batch_size
            end_i = start_i + batch_size
            xb = X[start_i:end_i]
            yb = y[start_i:end_i]

            print(xb)

            # Calculating hypothesis/prediction.
            y_hat = sigmoid(np.dot(xb, weights) + bias)

            # Getting the gradients of loss w.r.t parameters.
            dw, db = gradients(xb, yb, y_hat)

            # Updating the parameters.
            weights -= learning_rate * dw
            bias -= learning_rate * db

        # Calculating loss and appending it in the list.
        l = loss(sigmoid(np.dot(X, weights) + bias))
        losses.append(l)

    # returning weights, bias and losses(List).
    return weights, bias, losses

# Make the predictions
def predict(X, w, b):
    # X Input.

    # Calculating presictions/y_hat.
    preds = sigmoid(np.dot(X, w) + b)

    # Empty List to store predictions.
    pred_class = []

    # Delete the following two lines and replace it with your own
    for i in preds:
        pred_class.append(0)

    # if y_hat >= 0.5 round up to 1
    # if y_hat < 0.5 round down to 0
    for i in preds:
        if i >= 0.5:
            pred_class.append(1)
        else:
            pred_class.append(0)

    return np.array(pred_class)

# Obtain the accuracy
def accuracy(y, y_hat):
    # overwrite the accuracy value with your own code
    accuracy = 0
    # Ensure y and y_hat are numpy arrays
    y = np.array(y)
    y_hat = np.array(y_hat)

   # calculate accuracy
    correct_predictions = np.sum(y == y_hat)
    accuracy = correct_predictions / len(y)
    return accuracy

# Output the plot
def plot_decision_boundary(X, w, b):
    # X Inputs
    # w weights
    # b bias

    fig = plt.figure(figsize=(10, 8))
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "g^")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
    plt.xlim([-2, 2])
    plt.ylim([0, 2.2])
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title('Decision Boundary')

    # The Line is y=mx+c
    # So, Equate mx+c = w.X + b
    # Solving we find m and c
    x1 = [min(X[:, 0]), max(X[:, 0])]

    if (w[1] != 0):
        m = -w[0] / w[1]
        c = -b / w[1]
        x2 = m * x1 + c
        plt.plot(x1, x2, 'y-')

    plt.show()

    if (w[1] != 0):
        m = -w[0] / w[1]
        c = -b / w[1]
        x2 = m * x1 + c
        plt.plot(x1, x2, 'y-')

    plt.show()

# Evaluate the algorithm
filename = 'breast_cancer_data.csv'
dataset = load_csv(filename, skip=True)

diagnosis_column_to_number(dataset, 2)

X_train_data = extract_only_x_data(dataset)
y_train_data = extract_only_y_data(dataset)

X = np.array(X_train_data)
y = np.array(y_train_data)


# Training
w, b, l = train(X, y, batch_size=100, epochs=1000, learning_rate=0.01)
# Plotting Decision Boundary
plot_decision_boundary(X, w, b)

accuracy(y, y_hat=predict(X, w, b))