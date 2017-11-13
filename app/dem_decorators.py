


""" Some useful decorators for tensorboard reporting """


from functools import wraps
import tensorflow as tf


def tfNamespace(func):
    """ Tensorboard uses tf.name_scope to tag elements. By creating a decorator each function can
        be added to the Summary by name.""" 
    @wraps(func)
    def wrapper(*args, **kwargs):
        print func.__name__
        tf.name_scope(func.__name__)
        print tf.name_scope(func.__name__) 
        return func(*args, **kwargs)
    return wrapper




