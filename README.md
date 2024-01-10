# Back Propagation for Exam Results Prediction

This repository contains a Python script that implements a back propagation algorithm to train a neural network that predicts exam results from a school. The script was developed as part of a career subject "Artificial Intelligence and Expert Systems" from the Universidad Nacional de Misiones.

## Walkthrough

The script takes as input a CSV file with the following columns: student ID, gender, race/ethnicity, parental level of education, lunch, test preparation course, math score, reading score and writing score. This CSV file was proportioned by the university and for ethical reasons, can't be uploaded here. The script preprocesses the data by encoding the categorical variables and normalizing the numerical variables. Then, it splits the data into training and validation sets.

The script defines a neural network with one hidden layer of 3 neurons and an output layer of 2 neurons. The output layer represents the predicted grades for math, reading and writing, respectively. The script uses the mean squared error as the loss function and the sigmoid function as the activation function for both layers.

The script trains the neural network using the back propagation algorithm, which consists of two steps: forward propagation and backward propagation. In forward propagation, the script calculates the output of the network for each input and compares it with the actual output to compute the loss. In backward propagation, the script updates the weights and biases of the network by applying the gradient descent method, which moves them in the opposite direction of the gradient of the loss function.

The script repeats these steps for a number of epochs (iterations) until the loss converges to a minimum value or a maximum number of epochs is reached. The script also evaluates the performance of the network on the validation set after each epoch and prints the training and validation losses.

## Results

The script produces a plot that shows how the training and validation losses change over the epochs. The plot can be seen below:

![plot](plot.png)

The plot shows that both losses decrease as the epochs increase, indicating that the network is learning from the data. However, after around 50 epochs, the validation loss starts to increase slightly, while the training loss continues to decrease. This suggests that the network is overfitting to the training data and losing its ability to generalize to new data.

To prevent overfitting, one possible solution is to apply regularization techniques, such as dropout or weight decay, which reduce the complexity of the network and make it more robust to noise. Another possible solution is to use early stopping, which stops the training when the validation loss stops improving or starts worsening.

## Conclusion

The script demonstrates how to implement a back propagation algorithm to train a neural network that predicts exam results from a school. The script achieves a reasonable performance on both training and validation sets, but shows signs of overfitting after some epochs. The script can be improved by applying regularization or early stopping techniques to prevent overfitting and enhance generalization.
