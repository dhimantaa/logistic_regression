from function import *
from .propogation import optimize
from .initialize import initialize_with_zeros


def predict(weight, intercept, x_vector):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """

    m = x_vector.shape[1]
    y_prediction = np.zeros((1, m))
    weight = weight.reshape(x_vector.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    yhat = sigmoid(np.dot(weight.T, x_vector) + intercept)
    for i in range(yhat.shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if yhat[0][i] > 0.5:
            y_prediction[0][i] = 1
        else:
            y_prediction[0][i] = 0

    assert (y_prediction.shape == (1, m))

    return y_prediction


class Logistic(object):
    """
    This class provides the flexibility to run
    logistic regression to your data set
    """

    def __init__(self, *args, **kwargs):
        """
        Initializing the model parameter
        :param args:
        :param kwargs:
            X_train,
            Y_train,
            X_test,
            Y_test,
            num_iterations = 2000,
            learning_rate = 0.5
        """
        # Initializing the test & training set
        self._x_train = kwargs['X_train']
        self._y_train = kwargs['Y_train']
        self._x_test = kwargs['X_test']
        self._y_test = kwargs['Y_test']

        self.num_iteration = kwargs['num_iteration']
        self.learning_rate = kwargs['learning_rate']

    def fit(self):
        """
        function will fit the model with initialized parameter
        :return:
            costs,
            y_prediction_test,
            y_prediction_train,
            weight,
            intercept,
            self.learning_rate,
            self.num_iteration
        """
        # initialize parameters with zeros (≈ 1 line of code)
        weight, intercept = initialize_with_zeros(self._x_train.shape[0])

        # Gradient descent (≈ 1 line of code)
        parameters, grads, costs = optimize(weight,
                                            intercept,
                                            self._x_train,
                                            self._y_train,
                                            self.num_iteration,
                                            self.learning_rate
                                            )

        # Retrieve parameters w and b from dictionary "parameters"
        weight = parameters["w"]
        intercept = parameters["b"]

        # Predict test/train set examples (≈ 2 lines of code)
        y_prediction_test = predict(weight, intercept, self._x_test)
        y_prediction_train = predict(weight, intercept, self._x_train)

        # Print train/test Errors
        print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - self._y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - self._x_test)) * 100))

        return {"costs": costs,
                "Y_prediction_test": y_prediction_test,
                "Y_prediction_train": y_prediction_train,
                "w": weight,
                "b": intercept,
                "learning_rate": self.learning_rate,
                "num_iterations": self.num_iteration}
