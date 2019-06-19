from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(num_train):
        # X_i is (D, ) vector, W is (D, C) vector
        scores = X[i].dot(W) # (1, C) vector
        correct_class_score = scores[y[i]]
        
        # L_i = -log(exp(s_yi) / sum(exp(s_j)))
        exp_scores = np.exp(scores)
        sum_scores = np.sum(exp_scores)
        loss += (-1) * np.log(exp_scores[y[i]] / sum_scores)
        dW[:, y[i]] += X[i] * (-1 + (exp_scores[y[i]] / sum_scores))
        for j in range(num_classes):
            if j == y[i]:
                continue # Don't overwrite dW[y[i]]
            # dW for incorrect classes
            dW[:, j] += (exp_scores[j] / sum_scores) * X[i]
        
    # Divide by num_train to get average over individual losses
    loss /= num_train
    dW /= num_train
    
    # Add regularization term
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = X.dot(W) # N x C
    scores = np.exp(scores)
    sum_scores = np.sum(scores, axis=1)
    loss += (-1) * np.sum(np.log(scores[list(range(num_train)), y] / sum_scores))
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    s = np.divide(scores, sum_scores.reshape(num_train, 1))
    s[list(range(num_train)), y] = (-1 + scores[list(range(num_train)), y] / sum_scores)
    dW = X.T.dot(s)
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
