import json
import os

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE


class AdalineSGD:
    def __init__(self, learning_rate=0.01, num_iterations=100, random_seed=1):
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.num_iterations = num_iterations

    def fit(self, training_data, training_labels):
        if len(training_data.shape) == 1:
            training_data = training_data.reshape(-1, 1)

        # Initialize weights to zero
        self.weights = np.zeros(1 + training_data.shape[1])

        for _ in range(self.num_iterations):
            # Shuffle the data
            shuffled_indices = np.random.permutation(len(training_data))
            for i in shuffled_indices:
                net_input = self.calculate_net_input(training_data[i])
                output = self.activation_function(net_input)
                errors = (training_labels[i] - output)
                self.weights[1:] += self.learning_rate * training_data[i] * errors
                self.weights[0] += self.learning_rate * errors

        return self

    def calculate_net_input(self, features):
        # Calculate the net input by taking the dot product of features and weights, and adding the bias weight
        return np.dot(features, self.weights[1:]) + self.weights[0]

    def activation_function(self, x):
        # The activation function is the identity function, so the output is the same as the input
        return x

    def predict(self, features):
        # Predict the class labels by calculating the net input, applying the activation function,
        # and assigning 1 if the output is greater than or equal to 0, otherwise -1
        return np.where(self.activation_function(self.calculate_net_input(features)) >= 0.0, 1, -1)



# def balance_data(features, labels):
#     sm = SMOTE(random_state=42)
#     return sm.fit_resample(features, labels)
#

def classify_letters(data, letter1, letter2):
    # Filter the data to include only the specified letters
    filtered_data = [x for x in data if x[0] in [letter1, letter2]]
    if not filtered_data:
        print(f"No data found for labels {letter1} and {letter2}")
        return

    features, labels = zip(*[(item[1:], item[0]) for item in filtered_data])

    features = np.array(features)
    labels = np.array(labels)

    # Convert labels to binary values (1 or -1) based on the specified letters
    labels = np.where(labels == letter1, 1, -1)

    # Split the data into 80% for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Use K-Fold cross-validation for training and evaluating the model
    kf = KFold(n_splits=5)

    learning_rates = [0.0001, 0.001, 0.01]  # different learning rates to try
    num_iterations_list = [100, 500, 1000]  # different numbers of iterations to try

    best_accuracy = 0
    best_lr = None
    best_num_iterations = None

    # Try each combination of hyperparameters
    for lr in learning_rates:
        for num_iterations in num_iterations_list:
            accuracies = []

            for train_index, test_index in kf.split(X_train):
                X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
                y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

                # Create an instance of the AdalineSGD class with the current hyperparameters
                model = AdalineSGD(learning_rate=lr, num_iterations=num_iterations)

                # Train the model using the training data from the fold
                model.fit(X_train_fold, y_train_fold)

                # Predict the labels of the test data from the fold
                predictions = model.predict(X_test_fold)

                # Calculate the accuracy of the model by comparing the predicted labels to the true labels
                accuracy = accuracy_score(y_test_fold, predictions)
                accuracies.append(accuracy)

            # If this combination of hyperparameters achieved the highest average accuracy so far, store it
            if np.mean(accuracies) > best_accuracy:
                best_accuracy = np.mean(accuracies)
                best_lr = lr
                best_num_iterations = num_iterations

    # Now we know the best hyperparameters, train the final model on the entire training set
    model = AdalineSGD(learning_rate=best_lr, num_iterations=best_num_iterations)
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    predictions = model.predict(X_test)

    return accuracy_score(y_test, predictions)


def load_data(dirname: str) -> list[np.array]:
    data = []

    files = [os.path.join(dirname, file) for file in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, file))]

    print("Loading", len(files), "files as data")
    for file in files:
        try:
            with open(file) as f:
                content = f.read()
        except:
            print("Error in: ", file)
            continue

        content = content.replace("(", "[").replace(")", "]")

        for line in content.split('\n'):
            if ('[' in line) and (']' in line):
                try:
                    arr = json.loads(line)
                    if isinstance(arr, list):
                        img = np.array(arr)
                        if img.size != 101 or ((img[1:] == -1) | (img[1:] == 1)).sum() != 100:
                            print("Error in: ", file)
                        else:
                            data.append(img)
                except:
                    print("Error in: ", file)

    print(len(data), " images loaded as data")

    return data


if __name__ == '__main__':
    # Directory containing the files
    directory = 'C:/Cws/Adaline/lettersVecFiles'
    # List all files in the directory and its subdirectories
    data = []

    data.extend(load_data("vec"))
    data.extend(load_data("tvec"))

    # Classify the letters and calculate the average accuracies for different pairs of letters
    print("Average accuracy for bet vs mem: ", classify_letters(data, 1, 2))
    print("Average accuracy for lamed vs bet: ", classify_letters(data, 3, 1))
    print("Average accuracy for lamed vs mem: ", classify_letters(data, 3, 2))
