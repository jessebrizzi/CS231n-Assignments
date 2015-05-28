import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None

  ## reshape the input into rows leaving the original x intact
  # this reshapes x into N by prod(d_1, ..., d_k) shape
  temp = x.reshape(x.shape[0], -1)

  # find the scores
  out  = np.dot(temp, w) + b

  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None

  ## reshape the input into rows leaving the original x intact
  # this reshapes x into N by prod(d_1, ..., d_k) shape
  temp = x.reshape(x.shape[0], -1)

  dw = np.dot(temp.T, dout) # the x data dotted with the returned derivative of this nodes output
  dx = np.dot(dout, w.T).reshape(x.shape) # need to return it back to the original x shape
  db = np.sum(dout, axis=0) # sum each row of the returned derivative of this nodes output
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None

  # fairly simple, just the ReLU function f(x) = max(0, x)
  out = np.maximum(0, x)

  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache

  # the ReLU function f(x) = max(0, x)
  # f'(x) = 1(x > 0)x
  dx = dout
  dx[x <= 0] = 0

  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None

  # x: Input data of shape (N, C, H, W)
  N, C, H, W = x.shape
  # w: Filter weights of shape (F, C, HH, WW)
  F, CC, HH, WW = w.shape

  # 'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
  S = conv_param['stride']
  # 'pad': The number of pixels that will be used to zero-pad the input.
  P = conv_param['pad']

  # We can compute the spatial size of the output volume as a function of the input volume size (W), the receptive
  # field size of the Conv Layer neurons (F), the stride with which they are applied (S), and the amount of zero
  # padding used (P) on the border. You can convince yourself that the correct formula for calculating how many
  # neurons "fit" is given by (W-F+2P)/S+1. If this number is not an integer, then the strides are set incorrectly
  # and the neurons cannot be tiled so that they "fit" across the input volume neatly, in a symmetric way.
  H_out = (H - HH + 2*P)/S + 1
  W_out = (W - WW + 2*P)/S + 1
  out = np.zeros((N, F, H_out, W_out))

  # pad the input image with 0's
  x_pad = np.pad(x, [(0, 0), (0, 0), (P, P), (P, P)], 'constant')

  # for each training sample
  for n in xrange(N):
    # for each neuron
    for f in xrange(F):
      # for the stride across the image height
      for h_out in xrange(H_out):
        # for the stride across the image weight
        for w_out in xrange(W_out):
          # extract the window from the stride
          window = x_pad[n, :, h_out*S : h_out*S+HH, w_out*S : w_out*S+WW]
          # perform the convolution on the window with the weight(kernel/parameters)
          out[n, f, h_out, w_out] = np.sum(window * w[f]) + b[f]

  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  x, w, b, conv_param = cache

  # dout: Upstream derivatives.
  dN, dF, dH, dW = dout.shape
  # x: Input data of shape (N, C, H, W)
  N, C, H, W = x.shape
  # w: Filter weights of shape (F, C, HH, WW)
  F, CC, HH, WW = w.shape

  # 'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
  S = conv_param['stride']
  # 'pad': The number of pixels that will be used to zero-pad the input.
  P = conv_param['pad']

  # init the sizes of the outputs
  dx = np.zeros(x.shape)
  dw = np.zeros(w.shape)
  db = np.zeros(b.shape)

  # pad the input image with 0's
  x_pad = np.pad(x, [(0,0), (0,0), (P, P), (P, P)], 'constant')
  # pad the input image with 0's
  dx_pad = np.pad(dx, [(0,0), (0,0), (P, P), (P, P)], 'constant')

  # for each training sample
  for n in xrange(dN):
    # for each neuron
    for f in xrange(dF):
      # for the stride across the image height
      for dh in xrange(dH):
        # for the stride across the image weight
        for i in xrange(dW):
          # extract the window from the stride
          window = x_pad[n, :, dh*S : dh*S+HH, i*S : i*S+WW]
          # the derivative of w is the input window values times the returned derivative from up one level.
          dw[f] += window * dout[n, f, dh, i]
          # the derivative of x is the input weights times the returned derivative from up one level.
          dx_pad[n, :, dh*S : dh*S+HH, i*S : i*S+WW] += w[f] * dout[n, f, dh, i]

  # remove the padding and assign to the output for dx
  dx = dx_pad[:, :, P:P+H, P:P+W]
  # summing along multiple dimensions
  db = np.sum(np.sum(np.sum(dout, axis=0),axis=1),axis=1)

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None

  N, C, H, W = x.shape

  # Accepts a volume of size W1*H1*D1
  # Requires three hyperparameters:
    # their spatial extent F,
    # the stride S,

  FH = pool_param['pool_height']
  FW = pool_param['pool_width']
  S = pool_param['stride']

  # Produces a volume of size W2*H2*D2 where:
    # W2=(W1-F)/S+1
    # H2=(H1-F)/S+1
    # D2=D1

  H_out = (H - FH)/S + 1
  W_out = (W - FW)/S + 1
  out = np.zeros([N, C, H_out, W_out])

  # for each training sample
  for n in xrange(N):
    # for each depth/color channel
    for c in xrange(C):
      # for the ouput pooled height
      for h_out in xrange(H_out):
        # for the output pooled width
        for w_out in xrange(W_out):
          # get the pool window
          window = x[n, c, h_out*S : h_out*S+FH, w_out*S : w_out*S+FW]
          # pull the max value from the pool window
          out[n, c, h_out, w_out] = np.max(window)

  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None

  x, pool_param = cache
  N, C, H, W = x.shape

  # Accepts a volume of size W1*H1*D1
  # Requires three hyperparameters:
    # their spatial extent F,
    # the stride S,

  FH = pool_param['pool_height']
  FW = pool_param['pool_width']
  S = pool_param['stride']

  # Produces a volume of size W2*H2*D2 where:
    # W2=(W1-F)/S+1
    # H2=(H1-F)/S+1
    # D2=D1

  H_out = (H - FH)/S + 1
  W_out = (W - FW)/S + 1

  # the derivative of x will have the same size.
  dx = np.zeros(x.shape)

  # for each training example
  for n in xrange(N):
    # for each depth/color channel
    for c in xrange(C):
      # for the output pooled height
      for h_out in xrange(H_out):
        # for the output pooled width
        for w_out in xrange(W_out):
          # get the pool window
          window = x[n, c, h_out*S : h_out*S+FH, w_out*S : w_out*S+FW]
          # mask of the window where 1 = max_value, else 0
          mask = (window == np.max(window))
          # multiply the returned derivative from the level up by a binary mask of the window where 1 = max_value, else 0
          dx[n, c, h_out*S : h_out*S+FH, w_out*S : w_out*S+FW] +=  mask * dout[n, c, h_out, w_out]

  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

