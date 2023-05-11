import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


class AdalineSGD:
    def __init__(self, learning_rate=0.01, num_iterations=10, random_seed=1, epochs=50):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed
        self.num_iterations = num_iterations

    def fit(self, training_data, training_labels):
        if len(training_data.shape) == 1:
            training_data = training_data.reshape(-1, 1)

        random_generator = np.random.RandomState(self.random_seed)
        # Initialize weights randomly from a normal distribution with mean 0 and standard deviation 0.01
        self.weights = random_generator.normal(loc=0.0, scale=0.01, size=1 + training_data.shape[1])

        for _ in range(self.num_iterations):
            # Calculate the net input (weighted sum) of the features
            net_input = self.calculate_net_input(training_data)
            # Apply the activation function to the net input (identity function in this case)
            output = self.activation_function(net_input)
            # Calculate the errors as the difference between the training labels and the predicted output
            errors = (training_labels - output)
            # Update the weights using Stochastic Gradient Descent (SGD)
            self.weights[1:] += self.learning_rate * training_data.T.dot(errors)
            self.weights[0] += self.learning_rate * errors.sum()

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


def classify_letters(data, letter1, letter2):
    # Filter the data to include only the specified letters
    filtered_data = [x for x in data if x[1] in [letter1, letter2]]

    if not filtered_data:
        print(f"No data found for labels {letter1} and {letter2}")
        return

    features, labels = zip(*[(feature, label) for feature, label in filtered_data])

    features = np.array(features)
    labels = np.array(labels)

    # Convert labels to binary values (1 or -1) based on the specified letters
    labels = np.where(labels == letter1, 1, -1)

    # Create an instance of the AdalineSGD class
    model = AdalineSGD()

    # Use K-Fold cross-validation for training and evaluating the model
    kf = KFold(n_splits=5)

    accuracies = []
    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Train the model using the training data
        model.fit(X_train, y_train)

        # Predict the labels of the test data
        # Predict the labels of the test data
        predictions = model.predict(X_test)

        # Calculate the accuracy of the model by comparing the predicted labels to the true labels
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)

    # Return the average accuracy across all cross-validation folds
    return np.mean(accuracies)


def parse_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            # Parse each line of the file to extract the label and features
            elements = line.strip().replace('(', '').replace(')', '').split(',')
            elements = [int(e) for e in elements]
            label = elements[0]
            features = elements[1:]
            # Append the features and label as a tuple to the data list
            data.append((features, label))
    return data


if __name__ == '__main__':
    # Parse the data from the file
    data = parse_data("result.txt")

    # Classify the letters and calculate the average accuracies for different pairs of letters
    print("Average accuracy for bet vs mem: ", classify_letters(data, 1, 2))
    print("Average accuracy for lamed vs bet: ", classify_letters(data, 3, 1))
    print("Average accuracy for lamed vs mem: ", classify_letters(data, 3, 2))
