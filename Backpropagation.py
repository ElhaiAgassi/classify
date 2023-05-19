import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

import myUtils


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.weights_1 = np.random.randn(input_size, hidden_size)
        self.weights_2 = np.random.randn(hidden_size, output_size)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        self.hidden = self.sigmoid(np.dot(X, self.weights_1))
        output = self.sigmoid(np.dot(self.hidden, self.weights_2))
        return output

    def backpropagation(self, X, y, output):
        output_error = y - output
        hidden_error = output_error.dot(self.weights_2.T)

        self.weights_2 += self.hidden.T.dot(output_error * self.sigmoid_derivative(output)) * self.learning_rate
        self.weights_1 += X.T.dot(hidden_error * self.sigmoid_derivative(self.hidden)) * self.learning_rate

    def train(self, X, y):
        output = self.feedforward(X)
        self.backpropagation(X, y, output)

    def predict_classes(self, X):
        return self.feedforward(X) > 0.5

    def score(self, X, y):
        return accuracy_score(y, self.predict_classes(X))


def classify_letters_nn(data, letter1, letter2):
    X_train, X_test, y_train, y_test = myUtils.prepare_data(data, letter1, letter2, "backpropagation")
    if X_train is None:  # if no data was found
        return

    # ensure y_train and y_test are 2D arrays
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # define input, hidden, and output sizes based on your data
    input_size = X_train.shape[1]
    hidden_size = 80
    output_size = 1

    learning_rate = 0.001
    epochs = 2000

    # Split the training set using KFold
    kf = KFold(n_splits=5)

    fold_accuracies = []
    table_data = []
    for fold, (train_index, test_index) in enumerate(kf.split(X_train)):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

        # Train the model for the given number of epochs
        for epoch in range(epochs):
            nn.train(X_train_fold, y_train_fold)

        # Test the model on the fold and compute accuracy
        fold_accuracy = nn.score(X_test_fold, y_test_fold)
        fold_accuracies.append(fold_accuracy)

        mean_accuracy = np.mean(fold_accuracies[:fold + 1])
        std_dev_accuracy = np.std(fold_accuracies[:fold + 1])

        table_data.append([fold + 1, fold_accuracy, mean_accuracy, std_dev_accuracy])

    # Initialize and train the model on the full training set
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    for epoch in range(epochs):
        nn.train(X_train, y_train)

    # Test the model on the full test set and compute accuracy
    mean_accuracy = np.mean(fold_accuracies)
    total_accuracy = nn.score(X_test, y_test)

    print(f"\nClassification Report for Letters {letter1} and {letter2}:")
    print(tabulate(table_data, headers=['Fold', 'Accuracy', 'Mean Accuracy', 'Std Dev'], floatfmt=".4f",
                   tablefmt='fancy_grid'))

    print(f"Mean Cross-Validation Accuracy: {mean_accuracy:.4f}\n")

    return total_accuracy


if __name__ == '__main__':
    data = []

    data.extend(myUtils.load_data("vec"))
    data.extend(myUtils.load_data("tvec"))

    print("Total accuracy for bet vs mem: ", classify_letters_nn(data, 1, 2))
    print("Total accuracy for lamed vs bet: ", classify_letters_nn(data, 3, 1))
    print("Total accuracy for lamed vs mem: ", classify_letters_nn(data, 3, 2))
