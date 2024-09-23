from math import sqrt
from matplotlib import pyplot as plot
from random import seed
from random import randrange
from csv import reader

# Load the csv file
def load_csv(filename, skip=False):
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

# Convert any string column to a float coulm.
def string_column_to_float(dataset, column):
    for row in dataset:
        # The strip() function remove white space
        # then convert the data into a decimal number (float)
        # and overwrite the original data

        row[column] = float(row[column].strip())

# Calculate the mean value of a list of numbers.
def mean(values):
    mean_results = 0.0

    # Sum all the values and then divide number of values
    mean_results = sum(values) / float(len(values))

    return mean_results

# Calculate a regularisation value for the parameter.
def regularisation(parameter, lambda_value=0.01):
    parameter = parameter * (1 - lambda_value)

    return parameter

# Calculate least squares between x and y.
def leastSquares(dataset):
    x = list()
    y = list()

    for row in dataset:
        x.append(row[0])

    for row in dataset:
        y.append(row[1])

    b0 = 0
    b1 = 0

    # using the formula to calculate the b1 and b0
    numerator = 0
    denominator = 0

    x_mean = mean(x)
    y_mean = mean(y)

    for i in range(len(x)):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        denominator += (x[i] - x_mean) ** 2

    b1 = numerator / denominator
    b0 = y_mean - b1 * x_mean

    return [b0, b1]

# Calculate root mean squared error.
def root_mean_square_error(actual, predicted):
    rmse = 0.0
    sum_error = 0.0

    # Loops through the difference between the prediction
    # and the actual output
    # Then update the sum error
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)

    mean_error = sum_error / float(len(actual))
    rmse = sqrt(mean_error)
    return rmse

# Make Predictions.
def simple_linear_regression(train, test):
    predictions = list()
    b0, b1 = leastSquares(train)

    # Calculate the prediction (yhat)
    for row in test:
        yhat = b0 + b1 * row[0]
        predictions.append(yhat)

    return predictions

# Split the data into training and test sets.
def train_test_split(dataset, split):
    train = list()
    test = list(dataset)

    train_size = split * len(dataset)

    while len(train) < train_size:
        index = randrange(len(test))
        train.append(test.pop(index))

    return train, test

# Evaluate regression algorithm on training dataset.
def evaluate_simple_linear_regression(dataset, split=0):
    train, test = train_test_split(dataset, split)
    test_set = list()

    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)

    predicted = simple_linear_regression(train, test_set)

    actual = [row[-1] for row in test]

    rmse = root_mean_square_error(actual, predicted)

    return rmse

# Visualise the dataset.
def visualise_dataset(dataset):
    test_set = list()

    for row in dataset:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)

    sizes, prices = [], []
    for i in range(len(dataset)):
        sizes.append(dataset[i][0])
        prices.append(dataset[i][1])

    plot.figure()
    plot.plot(sizes, prices, 'x')
    plot.plot(test_set, simple_linear_regression(dataset, test_set))
    plot.xlabel('Fertility rate')
    plot.ylabel('Worker percent')
    plot.grid()
    plot.tight_layout()
    plot.show()

# Seed the random value
seed(1)

# Load and prepare data.
filename = 'fertility_rate-worker_percent.csv'
dataset = load_csv(filename, skip=True)

for i in range(len(dataset[0])):
    string_column_to_float(dataset, i)

# Evaluate algorithm.
split = 0.6
rmse = evaluate_simple_linear_regression(dataset, split)

print('Root Mean Square Error: %.3f' % rmse)
visualise_dataset(dataset)