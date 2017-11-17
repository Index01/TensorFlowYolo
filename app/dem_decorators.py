

""" 
    Some useful decorators for tensorboard reporting 
"""

from functools import wraps
import tensorflow as tf


def tfNamespace(func):
    """ Tensorboard uses tf.name_scope to tag elements. By creating a decorator each function can
        be added to the Summary by name.
        Right now this decorator only grabs the Weights, useful during convolution
    """ 
    @wraps(func)
    def wrapper(*args, **kwargs):
        with tf.name_scope(func.__name__):
            print func.__name__
            #TODO: extend this functonality to graph other tensor datatypes.
            try: 
                tf.summary.histogram('Weights', kwargs['W'])
            except KeyError:
                pass
            return func(*args, **kwargs) 
    return wrapper


def tfNamespaceWeight(func):
    """ TensorBoard namespace scope and summary for Weight values.""" 
    @wraps(func)
    def wrapper(*args, **kwargs):
        with tf.name_scope(func.__name__):
            print func.__name__
            ret = func(*args, **kwargs)
            tf.summary.histogram('Weight', ret)
            return ret
    return wrapper


def tfNamespaceBias(func):
    """ TensorBoard namespace scope and summary for Bias values.""" 
    @wraps(func)
    def wrapper(*args, **kwargs):
        with tf.name_scope(func.__name__):
            print func.__name__
            ret = func(*args, **kwargs)
            tf.summary.histogram('Biases', ret)
            return ret
    return wrapper


def tfNamespaceScalar(func):
    """ TensorBoard namespace scope and summary for Bias values.""" 
    @wraps(func)
    def wrapper(*args, **kwargs):
        with tf.name_scope(func.__name__):
            print func.__name__
            ret = func(*args, **kwargs)
            tf.summary.scalar(func.__name__, ret)
            return ret
    return wrapper




