import numpy as np


class ClassifierTrainer(object):
  """ The trainer class performs SGD with momentum on a cost function """
  def __init__(self):
    self.step_cache = {} # for storing velocities in momentum update

  def train(self, X, y, X_val, y_val, 
            model, loss_function, 
            reg=0.0,
            learning_rate=1e-2, momentum=0, learning_rate_decay=0.95,
            update='momentum', sample_batches=True,
            num_epochs=30, batch_size=100, acc_frequency=None,
            verbose=False):
    """
    Optimize the parameters of a model to minimize a loss function. We use
    training data X and y to compute the loss and gradients, and periodically
    check the accuracy on the validation set.

    Inputs:
    - X: Array of training data; each X[i] is a training sample.
    - y: Vector of training labels; y[i] gives the label for X[i].
    - X_val: Array of validation data
    - y_val: Vector of validation labels
    - model: Dictionary that maps parameter names to parameter values. Each
      parameter value is a numpy array.
    - loss_function: A function that can be called in the following ways:
      scores = loss_function(X, model, reg=reg)
      loss, grads = loss_function(X, model, y, reg=reg)
    - reg: Regularization strength. This will be passed to the loss function.
    - learning_rate: Initial learning rate to use.
    - momentum: Parameter to use for momentum updates.
    - learning_rate_decay: The learning rate is multiplied by this after each
      epoch.
    - update: The update rule to use. One of 'sgd', 'momentum', or 'rmsprop'.
    - sample_batches: If True, use a minibatch of data for each parameter update
      (stochastic gradient descent); if False, use the entire training set for
      each parameter update (gradient descent).
    - num_epochs: The number of epochs to take over the training data.
    - batch_size: The number of training samples to use at each iteration.
    - acc_frequency: If set to an integer, we compute the training and
      validation set error after every acc_frequency iterations.
    - verbose: If True, print status after each epoch.

    Returns a tuple of:
    - best_model: The model that got the highest validation accuracy during
      training.
    - loss_history: List containing the value of the loss function at each
      iteration.
    - train_acc_history: List storing the training set accuracy at each epoch.
    - val_acc_history: List storing the validation set accuracy at each epoch.
    """

    N = X.shape[0]

    if sample_batches:
      iterations_per_epoch = N / batch_size # using SGD
    else:
      iterations_per_epoch = 1 # using GD
    num_iters = num_epochs * iterations_per_epoch
    epoch = 0
    best_val_acc = 0.0
    best_model = {}
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    for it in xrange(num_iters):
      if it % 10 == 0:  print 'starting iteration ', it

      # get batch of data
      if sample_batches:
        batch_mask = np.random.choice(N, batch_size)
        X_batch = X[batch_mask]
        y_batch = y[batch_mask]
      else:
        # no SGD used, full gradient descent
        X_batch = X
        y_batch = y

      # evaluate cost and gradient
      cost, grads = loss_function(X_batch, model, y_batch, reg)
      loss_history.append(cost)

      # perform a parameter update
      for p in model:
        # compute the parameter step
        if update == 'sgd':
          dx = -learning_rate * grads[p]
        elif update == 'momentum':
          if not p in self.step_cache: 
            self.step_cache[p] = np.zeros(grads[p].shape)
          # Momentum update is another approach that almost always enjoys better converge rates on deep networks. This
          # update can be motivated from a physical perspective of the optimization problem. In particular, the loss
          # can be interpreted as a the height of a hilly terrain (and therefore also to the potential energy since
          # U=mgh and therefore U \propto h ). Initializing the parameters with random numbers is equivalent to setting a
          # particle with zero initial velocity at some location. The optimization process can then be seen as
          # equivalent to the process of simulating the parameter vector (i.e. a particle) as rolling on the landscape.

          # Since the force on the particle is related to the gradient of potential energy (i.e. F = -\DELTA*U ), the force
          # felt by the particle is precisely the (negative) gradient of the loss function. Moreover, F=ma so the
          # (negative) gradient is in this view proportional to the acceleration of the particle. Note that this is
          # different from the SGD update shown above, where the gradient directly integrates the position. Instead,
          # the physics view suggests an update in which the gradient only directly influences the velocity, which in
          # turn has an effect on the position:
          self.step_cache[p] = momentum * self.step_cache[p] - learning_rate * grads[p]
          dx = self.step_cache[p]
          # Here we see an introduction of a dx variable that is initialized at zero, and an additional hyperparameter
          # (momentum). As an unfortunate misnomer, this variable is in optimization referred to as momentum (its typical
          # value is about 0.9), but its physical meaning is more consistent with the coefficient of friction.
          # Effectively, this variable damps the velocity and reduces the kinetic energy of the system, or otherwise
          # the particle would never come to a stop at the bottom of a hill. When cross-validated, this parameter is
          # usually set to values such as [0.5, 0.9, 0.95, 0.99]. Similar to annealing schedules for learning rates
          # (discussed later, below), optimization can sometimes benefit a little from momentum schedules, where the
          # momentum is increased in later stages of learning. A typical setting is to start with momentum of about 0.5
          # and anneal it to 0.99 or so over multiple epochs.

        elif update == 'rmsprop':
          decay_rate = 0.99 # you could also make this an option
          if not p in self.step_cache: 
            self.step_cache[p] = np.zeros(grads[p].shape)

          # RMSprop. RMSprop is a very effective, but currently unpublished adaptive learning rate method. Amusingly,
          # everyone who uses this method in their work currently cites slide 29 of Lecture 6 of Geoff Hinton's
          # Coursera class. The RMSProp update adjusts the Adagrad method in a very simple way in an attempt to reduce
          # its aggressive, monotonically decreasing learning rate. In particular, it uses a moving average of squared
          # gradients instead, giving:
          self.step_cache[p] = decay_rate * self.step_cache[p] + (1 - decay_rate) * grads[p]**2
          dx = - learning_rate * grads[p] / np.sqrt(self.step_cache[p] + 1e-8)
          # Here, decay_rate is a hyperparameter and typical values are [0.9, 0.99, 0.999]. Notice that the x+= update
          # is identical to Adagrad, but the cache variable is a "leaky". Hence, RMSProp still modulates the learning
          # rate of each weight based on the magnitudes of its gradients, which has a beneficial equalizing effect,
          # but unlike Adagrad the updates do not get monotonically smaller.

        else:
          raise ValueError('Unrecognized update type "%s"' % update)

        # update the parameters
        model[p] += dx

      # every epoch perform an evaluation on the validation set
      first_it = (it == 0)
      epoch_end = (it + 1) % iterations_per_epoch == 0
      acc_check = (acc_frequency is not None and it % acc_frequency == 0)
      if first_it or epoch_end or acc_check:
        if it > 0 and epoch_end:
          # decay the learning rate
          learning_rate *= learning_rate_decay
          epoch += 1

        # evaluate train accuracy
        if N > 1000:
          train_mask = np.random.choice(N, 1000)
          X_train_subset = X[train_mask]
          y_train_subset = y[train_mask]
        else:
          X_train_subset = X
          y_train_subset = y
        scores_train = loss_function(X_train_subset, model)
        y_pred_train = np.argmax(scores_train, axis=1)
        train_acc = np.mean(y_pred_train == y_train_subset)
        train_acc_history.append(train_acc)

        # evaluate val accuracy
        scores_val = loss_function(X_val, model)
        y_pred_val = np.argmax(scores_val, axis=1)
        val_acc = np.mean(y_pred_val ==  y_val)
        val_acc_history.append(val_acc)
        
        # keep track of the best model based on validation accuracy
        if val_acc > best_val_acc:
          # make a copy of the model
          best_val_acc = val_acc
          best_model = {}
          for p in model:
            best_model[p] = model[p].copy()

        # print progress if needed
        if verbose:
          print ('Finished epoch %d / %d: cost %f, train: %f, val %f, lr %e'
                 % (epoch, num_epochs, cost, train_acc, val_acc, learning_rate))

    if verbose:
      print 'finished optimization. best validation accuracy: %f' % (best_val_acc, )
    # return the best model and the training history statistics
    return best_model, loss_history, train_acc_history, val_acc_history











