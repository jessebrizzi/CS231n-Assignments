import numpy as np
import matplotlib.pyplot as plt

def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model

def two_layer_net(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization the the weight matrices. The two layer net should use a ReLU
  nonlinearity after the first affine layer.

  The two layer net has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each
  class.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is not passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape

  # compute the forward pass
  scores = None

  ## FIRST LAYER
  # first multiply the data by the weights for the hidden layer -  "np.dot(X, W1)"
  # second add the bias for the hidden layer - "+ b1"
  # finally apply the ReLU activation function - "f(x) = max (0, x)"
  hidden1 = np.maximum(0, np.dot(X, W1) + b1)
  ## SECOND LAYER
  # first multiply the data by the weights for the output layer -  "np.dot(X, W1)"
  # second add the bias for the output layer - "+ b1"
  scores = np.dot(hidden1, W2) + b2
  
  # If the targets are not given then jump out, we're done
  if y is None:
    return scores

  # compute the loss
  loss = None

  # Given the array of scores we've computed above, we can compute the loss. First, the way to obtain the probabilities
  # is straight forward:
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

  # We now have an array probs of size [300 x 3], where each row now contains the class probabilities. In particular,
  # since we've normalized them every row now sums to one. We can now query for the log probabilities assigned to the
  # correct classes in each example:
  corect_logprobs = -np.log(probs[range(N),y])

  # The array correct_logprobs is a 1D array of just the probabilities assigned to the correct classes for each example.
  # The full loss is then the average of these log probabilities and the regularization loss:
  data_loss = np.sum(corect_logprobs)/N
  reg_loss = 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
  loss = data_loss + reg_loss

  # compute the gradients
  grads = {}

  # To get the gradient on the scores, which we call dscores, we proceed as follows:
  dscores = probs
  dscores[range(N),y] -= 1
  dscores /= N

  # backpropate the gradient to the parameters
  # first backprop into parameters W2 and b2
  grads['W2'] = np.dot(hidden1.T, dscores)
  grads['b2'] = np.sum(dscores, axis=0)
  # next backprop into hidden layer
  dhidden = np.dot(dscores, W2.T)
  # backprop the ReLU non-linearity
  dhidden[hidden1 <= 0] = 0
  # finally into W,b
  grads['W1'] = np.dot(X.T, dhidden)
  grads['b1'] = np.sum(dhidden, axis=0)

  # add regularization gradient contribution
  grads['W2'] += reg * W2
  grads['W1'] += reg * W1

  return loss, grads

