import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  # needed for calculations
  num_train = X.shape[1]

  for i in xrange(num_train):
    # calculate the scores for the current training example with the current weights
    scores = W.dot(X[:, i])
    # scale by the max for numerical stability
    scores -= np.max(scores)
    # calculate the loss
    loss += -scores[y[i]] + np.log(np.sum(np.exp(scores)))

    ## L' = -1_y + 1/(\sum_{}^{} e^f) * e^f
    # e^f
    scores = np.exp(scores)
    # 1/(\sum_{}^{} e^f)
    scores /= np.sum(scores)
    # -1_y
    scores[y[i]] -= 1

    # now scale it by the data
    # we need to use [:, np.newaxis] because when you make a X by 1 dimension slices in numpy the 1 dimension is null
    dW += scores[:, np.newaxis].dot(X[:, i][:, np.newaxis].T)


  # get the average loss
  loss /= num_train
  # get the average gradient
  dW /= num_train

  # regularize the loss function
  loss += 0.5 * reg * np.sum(W * W)

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """

  num_train = X.shape[1]

  # calculate the scores for the current training example with the current weights
  scores = W.dot(X)
  # scale by the max for numerical stability
  scores -= np.max(scores, axis = 0)
  # calculate the loss
  loss = np.sum(-scores[y, range(num_train)] + np.log(np.sum(np.exp(scores), axis = 0)))

  ## L' = -1_y + 1/(\sum_{}^{} e^f) * e^f
  # e^f
  scores = np.exp(scores)
  # 1/(\sum_{}^{} e^f)
  scores /= np.sum(scores,axis = 0)
  # -1_y
  scores[y, range(num_train)] -= 1
  # now we scale it by the data
  dW = scores.dot(X.T)

  # get the average loss
  loss /= num_train
  # get the average gradient
  dW /= num_train

  # regularize the loss function
  loss += 0.5 * reg * np.sum(W * W)

  return loss, dW
