import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


# Define the Adaline algorithm
class AdalineSGD:
    def __init__(self, learning_rate=0.01, num_iterations=10, random_seed=1, epochs=50):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed
        self.num_iterations = num_iterations

    def fit(self, training_data, training_labels):
        # Ensure training_data has two dimensions
        if len(training_data.shape) == 1:
            training_data = training_data.reshape(-1, 1)

        random_generator = np.random.RandomState(self.random_seed)
        self.weights = random_generator.normal(loc=0.0, scale=0.01, size=1 + training_data.shape[1])

        for _ in range(self.num_iterations):
            net_input = self.calculate_net_input(training_data)
            output = self.activation_function(net_input)
            errors = (training_labels - output)
            self.weights[1:] += self.learning_rate * training_data.T.dot(errors)
            self.weights[0] += self.learning_rate * errors.sum()

        return self

    def calculate_net_input(self, features):
        return np.dot(features, self.weights[1:]) + self.weights[0]

    def activation_function(self, x):
        return x

    def predict(self, features):
        return np.where(self.activation_function(self.calculate_net_input(features)) >= 0.0, 1, -1)


def classify_letters(data, letter1, letter2):
    # Filter out the data for the two letters we are interested in
    filtered_data = [x for x in data if x[1] in [letter1, letter2]]

    if not filtered_data:
        print(f"No data found for labels {letter1} and {letter2}")
        return

    # Separate the data into features and labels
    features, labels = zip(*[(feature, label) for feature, label in filtered_data])

    features = np.array(features)
    labels = np.array(labels)

    # Convert labels to 1 and -1 for the Adaline
    labels = np.where(labels == letter1, 1, -1)

def parse_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            # Remove parentheses and split the line into elements
            elements = line.strip().replace('(', '').replace(')', '').split(',')

            # Convert the elements to integers
            elements = [int(e) for e in elements]

            # First element is the label, the rest are the features
            label = elements[0]
            features = elements[1:]

            # Add the tuple to the data list
            data.append((features, label))

    return data


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = parse_data("result.txt")
    # Assuming 'data' is your list of tuples
    print("Average accuracy for bet vs mem: ", classify_letters(data, 1, 2))
    print("Average accuracy for lamed vs bet: ", classify_letters(data, 3, 1))
    print("Average accuracy for lamed vs mem: ", classify_letters(data, 3, 2))

