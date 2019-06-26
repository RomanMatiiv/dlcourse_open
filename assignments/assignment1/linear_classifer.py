import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops

    # for vector with shape (N)
    if predictions.ndim == 1:
        predictions -= np.max(predictions)
        predictions_exp = np.exp(predictions)

        probs = predictions_exp / predictions_exp.sum()

    # for matrix with shape (batch_size, N)
    elif predictions.ndim == 2:
        predictions -= predictions.max(axis=1)[:, None]
        predictions_exp = np.exp(predictions)

        probs = predictions_exp / predictions_exp.sum(axis=1)[:, None]

    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    if probs.ndim == 1:
        prob_true_class = probs[target_index]
        loss = -np.log(prob_true_class)

    elif probs.ndim == 2:
        target_index = np.array(target_index)
        target_index = target_index[:, None]
        prob_true_classes = np.take_along_axis(probs, target_index, axis=1)

        loss_for_each_elem = -np.log(prob_true_classes)
        loss = loss_for_each_elem.mean()

    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops

    if predictions.ndim!=target_index.ndim:
        target_index = target_index[:, None]
    predictions = predictions.copy()



    # for vector with shape (N)
    if predictions.ndim == 1:
        predictions -= np.max(predictions)
        predictions_exp = np.exp(predictions)

        predictions_exp_target = predictions_exp[target_index]

        softmax_target = predictions_exp_target/predictions_exp.sum()
        loss = -np.log(softmax_target)

        dprediction = predictions_exp/predictions_exp.sum()

        dprediction[target_index] = dprediction[target_index]-1

        dprediction

    # for matrix with shape (batch_size, N)
    elif predictions.ndim == 2:
        predictions -= predictions.max(axis=1)[:, None]
        predictions_exp = np.exp(predictions)

        predictions_exp_target = np.take_along_axis(predictions_exp, target_index, axis=1)

        softmax_target = predictions_exp_target/predictions_exp.sum(axis=1)[:,None]
        loss = -np.log(softmax_target).sum()

        dprediction = predictions_exp/predictions_exp.sum(axis=1)[:,None]
        dprediction_target = (predictions_exp_target / predictions_exp.sum(axis=1)[:, None]) - 1
        np.put_along_axis(dprediction, target_index, dprediction_target, axis=1)

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops

    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)

    # dW = W.copy()

    # dW = np.zeros_like(W)
    # for i in range(X.shape[0]):
    #     dW+=(X[:,None][i]*dprediction[i][:,None]).T

    dW = (X[:, None] * dprediction[:, :, None]).sum(axis=0).T


    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            raise Exception("Not implemented!")

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        raise Exception("Not implemented!")

        return y_pred


