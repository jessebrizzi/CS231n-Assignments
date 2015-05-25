import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    # calculate the scores for the current training example with the current weights
    scores = W.dot(X[:, i])
    # pull the correct classes score
    correct_class_score = scores[y[i]]
    # start the count for the gradient
    count = 0
    for j in xrange(num_classes):
      # skip the instance where the correct class
      if j == y[i]:
        continue
      # calculate the margin
      margin = scores[j] - correct_class_score + 1 # delta = 1
      # max(0, margin)
      if margin > 0:
        # count the number of classes that didn't meet the desired margin (and hence contributed to the loss function)
        count += 1
        # the other rows where j != yi the gradient
        dW[j, :] += X[:, i]
        # sum the loss over all data samples
        loss += margin
    # and then the data vector x_i scaled by this (count) number is the gradient.
    dW[y[i], :] += -1 * count * X[:, i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_train = X.shape[1]

  # calculate the scores for the current training example with the current weights
  scores = W.dot(X)
  # pull the correct classes score
  correct_scores = scores[y, range(num_train)]
  # calculate the margin
  margins = np.maximum(0, scores - correct_scores + 1)
  # skip the instance where the correct class
  margins[y, range(num_train)] = 0
  # sum the loss over all data samples
  loss = np.sum(margins)

  # count the number of classes that didn't meet the desired margin (and hence contributed to the loss function)
  margins[margins > 0] = 1
  count = np.sum(margins, axis=0)
  # and then the data vector xi scaled by this (count) number is the gradient.
  margins[y, range(num_train)] = -count

   # the other rows where j != yi the gradient
  dW = margins.dot(X.T)

  # get the average loss
  loss /= num_train
  # get the average gradient
  dW /= num_train

  # regularize the loss function
  loss += 0.5 * reg * np.sum(W * W)

  return loss, dW
