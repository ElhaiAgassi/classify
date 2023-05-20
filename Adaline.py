import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tabulate import tabulate

import myUtils


class AdalineSGD:
    def __init__(self, learning_rate=0.0001, num_iterations=1000, random_seed=1):
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.num_iterations = num_iterations

    def fit(self, training_data, training_labels):
        if len(training_data.shape) == 1:
            training_data = training_data.reshape(-1, 1)

        self.weights = np.zeros(1 + training_data.shape[1])

        for _ in range(self.num_iterations):
            shuffled_indices = np.random.permutation(len(training_data))
            for i in shuffled_indices:
                net_input = self.calculate_net_input(training_data[i])
                output = self.activation_function(net_input)
                errors = (training_labels[i] - output)
                self.weights[1:] += self.learning_rate * training_data[i] * errors
                self.weights[0] += self.learning_rate * errors

        return self

    def calculate_net_input(self, features):
        return np.dot(features, self.weights[1:]) + self.weights[0]

    def activation_function(self, x):
        return x

    def predict(self, features):
        return np.where(self.activation_function(self.calculate_net_input(features)) >= 0.0, 1, -1)

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        return accuracy_score(y_test, predictions)


def classify_letters(data, letter1, letter2):
    X_train, X_test, y_train, y_test = myUtils.prepare_data(data, letter1, letter2, "adaline")
    if X_train is None:  # if no data was found
        return
    kf = KFold(n_splits=5)

    fold_accuracies = []
    table_data = []
    for fold, (train_index, test_index) in enumerate(kf.split(X_train)):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        model = AdalineSGD()
        model.fit(X_train_fold, y_train_fold)

        fold_accuracy = model.score(X_test_fold, y_test_fold)
        fold_accuracies.append(fold_accuracy)

        mean_accuracy = np.mean(fold_accuracies[:fold + 1])
        std_dev_accuracy = np.std(fold_accuracies[:fold + 1])

        table_data.append([fold + 1, fold_accuracy, mean_accuracy, std_dev_accuracy])

    total_accuracy = model.score(X_test, y_test)

    print(f"\nClassification Report for Letters {letter1} and {letter2}:")
    print(tabulate(table_data, headers=['Fold', 'Accuracy', 'Mean Accuracy', 'Std Dev'], floatfmt=".4f",
                   tablefmt='fancy_grid'))

    print(f"Total Accuracy: {total_accuracy:.4f}\n")

    return total_accuracy


if __name__ == '__main__':
    data = []

    data.extend(myUtils.load_data("vec"))
    data.extend(myUtils.load_data("tvec"))

    print("Average accuracy for bet vs mem: ", classify_letters(data, 1, 2))
    print("Average accuracy for lamed vs bet: ", classify_letters(data, 3, 1))
    print("Average accuracy for lamed vs mem: ", classify_letters(data, 3, 2))
