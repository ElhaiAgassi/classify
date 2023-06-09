# Classification of Letters

This code is part of a task that involves classifying letters using different algorithms. In this part (Part A), we use
the Adaline (Adaptive Linear Neuron) algorithm with Stochastic Gradient Descent (SGD) for classification. The code
includes functions to parse the data from a file, perform K-Fold cross-validation, and evaluate the accuracy of the
model.

## Requirements

- numpy (v1.19.5)
- scikit-learn (v0.24.2)

## Code Explanation

The code consists of the following sections:

1. AdalineSGD Class:
    - This class implements the Adaline algorithm with Stochastic Gradient Descent for training.
    - The constructor initializes parameters such as the learning rate, number of iterations, random seed, and epochs.
    - The `fit` method trains the model using the provided training data and labels.
    - The `calculate_net_input` method calculates the net input of the model.
    - The `activation_function` method defines the activation function used in the model.
    - The `predict` method predicts the labels for the given features.

2. classify_letters Function:
    - This function performs the classification of letters based on the provided data, letter1, and letter2.
    - It filters the data to include only the specified letters.
    - It converts the labels to binary values (1 or -1).
    - It creates an instance of the AdalineSGD class.
    - It uses K-Fold cross-validation to train and evaluate the model's accuracy.
    - It returns the average accuracy across all folds.

3. parse_data Function:
    - This function parses the data from a file.
    - It reads each line of the file, removes unnecessary characters, and converts the elements to integers.
    - It returns a list of tuples, where each tuple contains the features and label of a data point.

4. Main Execution:
    - The code reads the data from the "result.txt" file using the `parse_data` function.
    - It performs the classification using the `classify_letters` function for different pairs of letters.
    - It prints the average accuracy for each pair of letters.

## How AdalineSGD Works

The Adaline algorithm is a linear classification algorithm that aims to adjust its weights to minimize the sum of
squared errors between the predicted and actual outputs. Here's a brief overview of how the AdalineSGD class works:

1. Initialization: The class is initialized with parameters like the learning rate, number of iterations, random seed,
   and epochs. Weights are randomly initialized using a normal distribution.

2. Training: The `fit` method is called to train the model. It calculates the net input by taking the dot product of the
   features and weights, applies the activation function, and computes the errors. The weights are updated using
   Stochastic Gradient Descent, where the learning rate is multiplied by the dot product of the training data and
   errors. The process is repeated for the specified number of iterations.

3. Net Input Calculation: The `calculate_net_input` method calculates the net input by taking the dot product of the
   features and weights, adding the bias weight.

4. Activation Function: The `activation_function` method defines the activation function used in Adaline, which is the
   identity function in this case.

5. Prediction: The `predict` method predicts the labels for the given features by calculating the net input, applying
   the activation function, and assigning 1 or -1 based on the result.

## Code Explanation: Part B - Neural Network with Backpropagation

In this part (Part B), we use a simple 2-layer Neural Network (with a single hidden layer) trained using the
Backpropagation algorithm for classification. The following sections of the code are present:

5. NeuralNetwork Class:
    - This class implements the 2-layer neural network.
    - The constructor initializes parameters such as the weights for the layers, learning rate.
    - The `sigmoid` method calculates the sigmoid function.
    - The `sigmoid_derivative` method calculates the derivative of the sigmoid function.
    - The `feedforward` method applies the feedforward process in the neural network.
    - The `backpropagation` method applies the backpropagation process to update the weights.
    - The `train` method applies both feedforward and backpropagation processes iteratively for the training data.
    - The `predict_classes` method predicts the class labels for the given features.
    - The `score` method calculates the accuracy of the model.

6. classify_letters_nn Function:
    - This function performs the classification of letters using the Neural Network.
    - It prepares the data (using the provided utility function), trains the neural network, and evaluates the model's
      accuracy using K-Fold cross-validation.
    - It returns the total accuracy across all test sets and folds.

7. Main Execution:
    - The main script loads data, performs the classification using the `classify_letters_nn` function for different
      pairs of letters, and prints the average accuracy for each pair of letters.

## How the NeuralNetwork Class Works

The NeuralNetwork class implements a simple 2-layer Neural Network (with a single hidden layer) and uses the
Backpropagation algorithm for training. Here's a brief overview:

1. Initialization: The class is initialized with weights randomly assigned for each layer.

2. Feedforward: The `feedforward` method performs a forward pass through the network. It calculates the dot product of
   the input and weights, applies the sigmoid function, and carries out this process for both layers.

3. Backpropagation: The `backpropagation` method performs a backward pass through the network. It calculates the error
   in the output, computes the derivative of the error, and updates the weights. The process is carried out for both
   layers.

4. Training: The `train` method performs the feedforward and backpropagation processes for a specified number of
   iterations (epochs).

5. Prediction: The `predict_classes` method performs a forward pass on the input data and returns class labels based on
   the output.

6. Scoring: The `score` method calculates the model's accuracy by comparing the predicted class labels with the actual
   labels.

## Running the Code

The instructions for running the code remain the same as in Part A. Note that the `classify_letters_nn` function needs
to be used for training and evaluation. This function trains the neural network using the provided training data,
performs k-fold cross-validation, and prints out the classification accuracy.

Here's how to run the updated script:

1. Ensure you have the required dependencies installed (numpy, scikit-learn, and tabulate).
2. Place the data file in the correct directory.
3. Run the script. It will load the data, train the model using the `classify_letters_nn` function, and print the
   average accuracy for each pair of letters.
4. The classification reports for each pair of letters are printed to the console, including individual fold accuracies,
   mean accuracy, and standard deviation for each k-fold cross-validation, and the overall accuracy on the test set.
