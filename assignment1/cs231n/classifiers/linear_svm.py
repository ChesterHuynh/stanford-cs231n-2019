from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) # s = f(x_i, W) = Wx_i
        correct_class_score = scores[y[i]] # y[i] is the label corresponding to ith input
        loss_contributors_count = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                # For incorrect classes
                # grad_{w_{j}}(L_i) = I(s_yi - s_j + 1 > 0) * * x_i
                dW[:,j] += X[i]
                loss_contributors_count += 1
        # For correct class
        # grad_{w_{y_i}}(L_i) = -1 * x_i * sum over all classes{I(s_yi - s_j + 1 > 0)}
        dW[:,y[i]] += (-1) * loss_contributors_count * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train # Average dW over number of training exmaples

    # Add regularization to the loss.
    loss += reg * np.sum(W * W) # \lambda * W^2
    dW += 2 * reg * W # take derivative of regularization term and include it in dW: \lambda * 2W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]
    s = X.dot(W) # (N, C) matrix
    # choose the correct class column in each row to form a column vector
    correct_class_scores = s[list(range(num_train)), y]
    correct_class_scores = correct_class_scores.reshape(num_train, -1)
    s += 1 - correct_class_scores # s_j - s_yi + 1
    s[list(range(num_train)), y] = 0 # Make sure correct scores don't contribute to loss
    loss = np.sum(np.fmax(s,0)) / num_train # Sum over the losses then divide by N
    loss += reg * np.sum(W * W) # Don't forget about the regularization term!
    
    loss_contributors = np.zeros(s.shape)
    # 0 if s_j - s_yi + 1 < 0, else 1
    loss_contributors[s > 0] = 1
    # Update for correct class in each row
    loss_contributors[np.arange(y.shape[0]), y] = -np.sum(loss_contributors, axis=1) 
    # Multiply by loss contributors by X to complete gradient calculation
    dW = X.T.dot(loss_contributors) / num_train
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
