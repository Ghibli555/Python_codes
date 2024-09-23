from random import seed
from random import randrange
import random
from csv import reader

from tabulate import tabulate

# Load the csv file.
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

# Print the csv file's content
def print_the_dataset(dataset, contents=True, length=True):
    if (contents):
        print(tabulate(dataset))

    if (length):
        print(len(dataset))

# Split the csv heart dataset into training and test datasets.
def train_test_split(dataset, split):

    # Create an empty list for the training set
    training = list()

    # Define the size of the training set
    train_size = split * len(dataset)

    # Copy the original dataset to
    test = list(dataset)

    # Loops only to the size of the training set
    while len(training) < train_size:
        # Obtain a random index from the dataset/test set
        index = randrange(len(test))

        # Populate the training set, by moving the data points from the
        # dataset/test set to the training set
        training.append(test.pop(index))

    # Return both the training set and test set
    return training, test

# Split the csv dataset into k folds for cross validation.
def k_fold_cross_validation(dataset, k):
    n = len(dataset)  # Length of the dataset
    fold_size = n // k  # Divide the length into smaller folds
    folds = []  # Empty list of folds

    # Shuffle the dataset
    shuffled_dataset = dataset.copy()
    random.shuffle(shuffled_dataset)

    for i in range(k):
        # Assign a start and end variables in respect to the fold size
        start = i * fold_size
        end = (i + 1) * fold_size if i != k - 1 else n

        # Generate all the test indices for the current fold
        test_indices = list(range(start, end))

        # Generate all the train indices for the all other folds
        train_indices = list(range(0, start)) + list(range(end, n))

        # Create a test set that is randomly populated via the test_indices
        test_set = [shuffled_dataset[j] for j in test_indices]

        # Create a train set that is randomly populated via the train_indices
        train_set = [shuffled_dataset[j] for j in train_indices]

        folds.append((train_set, test_set))

    return folds

# Seed the random value.
seed(1)

# Load the big heart csv file and split the data into training(80%) and test (20%) set
filename = 'big_heart.csv'

dataset = load_csv(filename, skip = True)
print_the_dataset(dataset)

training, test = train_test_split(dataset, 0.8)

print(len(training))

print(len(test))

# Load the dataset and assign the data into 5 folds.
k = 5  # Number of folds for cross-validation
folds = k_fold_cross_validation(dataset, k)

# Print the size of each fold
for i, fold in enumerate(folds):
    train_set, test_set = fold
    print(f"Fold {i+1}: Training set size: {len(train_set)}, Test set size: {len(test_set)}")


