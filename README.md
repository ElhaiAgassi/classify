# Classification of Letters

This code is part of a task that involves classifying letters using different algorithms. In this part (Part A), we use the Adaline (Adaptive Linear Neuron) algorithm with Stochastic Gradient Descent (SGD) for classification. The code includes functions to parse the data from a file, perform K-Fold cross-validation, and evaluate the accuracy of the model.

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

The Adaline algorithm is a linear classification algorithm that aims to adjust its weights to minimize the sum of squared errors between the predicted and actual outputs. Here's a brief overview of how the AdalineSGD class works:

1. Initialization: The class is initialized with parameters like the learning rate, number of iterations, random seed, and epochs. Weights are randomly initialized using a normal distribution.

2. Training: The `fit` method is called to train the model. It calculates the net input by taking the dot product of the features and weights, applies the activation function, and computes the errors. The weights are updated using Stochastic Gradient Descent, where the learning rate is multiplied by the dot product of the training data and errors. The process is repeated for the specified number of iterations.

3. Net Input Calculation: The `calculate_net_input` method calculates the net input by taking the dot product of the features and weights, adding the bias weight.

4. Activation Function: The `activation_function` method defines the activation function used in Adaline, which is the identity function in this case.

5. Prediction: The `predict` method predicts the labels for the given features by calculating the net input, applying the activation function, and assigning 1 or -1 based on the result.

## Running the Code

To run the code, follow these steps:

1. Make sure you have the required dependencies installed (numpy and scikit-learn).
2. Place the data file