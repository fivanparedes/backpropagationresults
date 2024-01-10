# Back Propagation for Exam Results Prediction

This repository contains a Python script that implements a back propagation algorithm to train a neural network that predicts exam results from a school. The script was developed as part of a career subject "Artificial Intelligence and Expert Systems" from the Universidad Nacional de Misiones.

## What is Backpropagation?
It is the essence of neural network training. It is a method that consists of adjusting the weights associated with the neurons in the network based on the error obtained in the previous iteration. The good adjustment of the weights allows us to The good adjustment of the weights allows us to reduce the error range making the model more reliable, increasing its generalization.
Back propagation is a short way of naming the back propagation of errors. It is a standard method of training neural networks which helps to calculate the gradient loss function for a weight by the chain rule. It efficiently computes one layer at a time, unlike native direct computation. Calculates the gradient, but does not define how the gradient is used. Generalizes the calculation in the delta rule.
Let us consider the following example diagram of back propagation neural network to understand the concept:

### How the back propagation algorithm works

1. X inputs, arriving via the preconnected route.
2. The input is modeled using real weights W. The weights are generally selected at random.
3. Calculate the output for each neuron from the input layer, to the hidden layers, to the output layer.
4. Calculate the error in the outputs: Error = Actual Output - Desired Output.
5. Travel back from the output layer to the hidden layer to adjust the weights so that the error decreases.

Keep repeating the process until the desired output is achieved.

## Walkthrough

The script takes as input a CSV file with the following columns: student ID, gender, race/ethnicity, parental level of education, lunch, test preparation course, math score, reading score and writing score. This CSV file was proportioned by the university and for ethical reasons, can't be uploaded here. The script preprocesses the data by encoding the categorical variables and normalizing the numerical variables. Then, it splits the data into training and validation sets.

The script defines a neural network with one hidden layer of 3 neurons and an output layer of 2 neurons. The output layer represents the predicted grades for math, reading and writing, respectively. The script uses the mean squared error as the loss function and the sigmoid function as the activation function for both layers.

The script trains the neural network using the back propagation algorithm, which consists of two steps: forward propagation and backward propagation. In forward propagation, the script calculates the output of the network for each input and compares it with the actual output to compute the loss. In backward propagation, the script updates the weights and biases of the network by applying the gradient descent method, which moves them in the opposite direction of the gradient of the loss function.

The script repeats these steps for a number of epochs (iterations) until the loss converges to a minimum value or a maximum number of epochs is reached. The script also evaluates the performance of the network on the validation set after each epoch and prints the training and validation losses.

## Classification tests
Once the training has been completed, we proceed to incorporate the test set, which consists of a matrix with several records of the same structure as the training set (hours, attendance, average of previous grades). The same process was applied as in the training stage, except for the part of the error calculation and the new weights.
Four test sets will be used, where each column is distributed as follows: [hours, attendance, average previous grades].

The goal is to perform the predictions with different sets representing different situations:
- Test set 1 represents a student who did not have even the slightest preparation on the exam, nor does he attend classes.
- Test set 2 represents the most diligent student who invested a good amount of preparation hours.
- Test set 3 is a student with a regular performance, who usually passes with just enough.
- Test set 4 is an average performer who does not fully understand the content.
  
SetTest1 = [0, 0.1, 0.2].
SetTest2 = [9, 1, 0.87]
SetTest3 = [3, 0.7, 0.63]
SetTest4 = [3, 0.7, 0.45]

### Ranking criteria
If the output obtained is greater than 0.55, the result classifies as "Pass", otherwise "Fail".

## Results

The following graph represents the evolution of the general absolute error (average of the absolute value of the error obtained in each iteration). It can be seen at around 9000 iterations, the error reaches a point where it does not change much in value, so we can conclude that the weights of the value does not change much, so we can conclude that the weights of the network have been sufficiently adjusted.

![Loss function](https://github.com/fivanparedes/backpropagationresults/blob/main/lossfn.png)

### Test execution
Learning factor is 0.05, the number of neurons in the hidden layer is 5 and the number of iterations is 10000. The initial weights are
completely random.

| TestSet   | N° of neurons  | Learning rate         | N° of iterations    | Output    | Result    |
| --------- | -------------- | --------------------- | ------------------- | --------- | --------- |
| 1         | 5              | 0.05                  | 10000               | 0.21011949| Fails     |
| 2         | 5              | 0.05                  | 10000               | 0.81986203| Passes    |
| 3         | 5              | 0.05                  | 10000               | 0.60465018| Fails     |
| 4         | 5              | 0.05                  | 10000               | 0.46288901| Passes    |


## Conclusion

A study was conducted to evaluate the behavior of a neural network with three inputs, a hidden layer of five neurons, and an output. The inputs were the number of hours of study as integers, the percentage of class attendance from 0 to 1, and the average of previous grades from 0 to 1. After 10000 iterations, a proper convergence and fitting of the results was achieved. The results obtained were a predicted final grade for the students, also from 0 to 1. Although the use of grades from 0 to 1 instead of 1 to 10 proved to be more beneficial for the creation of the model, there is still much room for improvement. To achieve this, attention must be paid to the quality of the input data. This means making sure that the data is accurate, complete, and up-to-date, as well as within a range of similar values so that it does not alter the model drastically, as it is well known that the backpropagation technique is sensitive to noisy data. This also means that there must be enough information for the model to learn and improve.

In addition, it is important to increase the importance of certain weights over others in order to others to make the model more accurate. This implies having to understand and study the input data in order to see which weights are more important to the model, and how to adjust importance to the model, and how to make the necessary adjustments to reinforce them, as in the case of attendance the model, and how to make the necessary adjustments to reinforce them, as in the case of class attendance and GPA. In our opinion, the hours of study should be more decisive. should be more decisive. With the appropriate use of these techniques, it is possible to the quality of the model and ensure that it is accurate and reliable. 
It is concluded that the neural network model performed adequately in predicting the final grade of the students. predicting the final grade of the students as it obtained an accuracy factor of 97.1%, which is accuracy factor of 97.1% was obtained, which we consider highly acceptable.
