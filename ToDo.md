# TODO List

## 1. Preprocessing Data

Data is raw, so we need to preprocess the data.

## 2. Argument of file

python file.py --acti sigmoid ReLU tanh --layer 24 24 24 --epochs 84 --loss binaryCrossentropy --batch_size 8 --learning_rate 0.0314

## 3. Activation function

You must also implement the softmax function on the output layer in order to
obtain the output as a probabilistic distribution

## 4. Evaluation during training (split the dataset)

While training, we have to evaluate our model with the validation part of our dataset

We have to display at each epochs the validation and training metrics

## 5. Display learning Curve

We will also implement two learning curve graphs displayed at the end of the
training phase (example : Loss and Accuracy)

## 6. Creating 3 program

• A program to separate the data set into two parts, one for training and the other for validation

- For the separate program you are allowed to use a seed to obtain a repeatable
  result, because a lot of random factors come into play (the weights and bias
  initialization for example)

• A training program

- The training program will use backpropagation and gradient descent to learn
  on the training dataset and will save the model (network topology and weights) at
  the end of its execution.

• A prediction program

- The prediction program will load the weights learned in the previous phase,
  perform a prediction on a given set (which will also be loaded), then evaluate it
  using the binary cross-entropy error function :\
  E = − 1/N (∑ n=1->N) [yn log pn + (1 − yn) log(1 − pn)]


## before correction

- add option to command line for early stop (boolean)
- add an optimizer technique to increase SGD
- make a .py to compare different type of activations functions
- add other metrics (f1 score/ recall ) etc etc
- save and load layers param with specific filename
- add .py for prediction
