
"""
   CNN model building blocks.

   Currently these modules are using constants which anticipate the MNist dataset as input.
"""

import tensorflow as tf
from dem_decorators import tfNamespace, tfNamespaceBias, tfNamespaceWeight, tfNamespaceScalar 


###########################################
### Model parameters below              ###
###########################################

def IO_setup():
    """ Inputs and Outputs """
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
    return x, y_
 

@tfNamespaceWeight
def W_shop(shape):
    """ Model weights"""
    W = tf.truncated_normal(shape, stddev=0.1, name='W')
    return tf.Variable(W)


@tfNamespaceBias
def b_shop(shape):
    """ Model biases"""
    b = tf.constant(0.1, shape=shape, name='b')
    return tf.Variable(b)


@tfNamespace
def convolve(x, W):
    """ Convolve the image with the weight """
    strides = [1, 1, 1, 1]
    #print x
    #print W
    return tf.nn.conv2d(input=x, 
		        filter=W,
		        strides=strides, 
		        padding='SAME', 
		        name="convolve") 

    
@tfNamespace
def pool(x):
    """ Pool the convolutions """
    strides = [1, 2, 2, 1]
    ksize = [1, 2, 2, 1]
    with tf.name_scope("pool"):
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding='SAME') 


###########################################
### Model setup below                   ###
###########################################


@tfNamespace
def create_conv_layer(x, weights, biases):
    """ Provide a flexible combination of convolution and pooling steps. 
        Args: an image, a list of weights, list of biases
    """
    with tf.name_scope("conv_layer"):
        W_conv = W_shop(weights)
        b_conv = b_shop(biases)
        
        # Lossy
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        
        # Relu
        h_conv = tf.nn.relu(convolve(x=x_image, W=W_conv) + b_conv) 
        return pool(h_conv)


@tfNamespace
def create_subsequent_conv_layer(previous_layer, weights, biases):
    """ The subsequent layers need a slightly different method """
    with tf.name_scope("second_conv_layer"):
        W_conv2 = W_shop(weights)
        b_conv2 = W_shop(biases)
    
        # Relu 
        h_conv2 = tf.nn.relu(convolve(previous_layer, W=W_conv2) + b_conv2) 
        return pool(h_conv2)


@tfNamespace
def create_connected_layer(layer_in, dim1):
    """ Create a fully connected layer with relu"""
    with tf.name_scope("connected_layer"):
        flat_shape = [-1, 7 * 7 * dim1] 
  
        flattened = tf.reshape(layer_in, flat_shape) 
        W_flat = W_shop([(7 * 7 * dim1), 1024]) 
        b_flat = b_shop([1024]) 

        return tf.nn.relu(tf.matmul(flattened, W_flat) + b_flat) 


@tfNamespace
def create_logits(drop_op):
    """ Accept a dense layer and return logits"""
    W_logs = W_shop([1024, 10]) 
    b_logs = b_shop([10]) 
    return tf.matmul(drop_op, W_logs) + b_logs
    #return tf.reshape(tf.matmul(drop_op, W_logs) + b_logs, [-1, 16, 16, 1])


@tfNamespace
def drop_neurons(fully_connected_layer, keep_prob):
    """ DDDDDdropout """
    #TODO: review keep_prob implementation 
#    keep_prob = tf.placeholder(tf.float32, shape=[])
#    keep_prob = tf.placeholder(tf.float32)
    cropped =tf.nn.dropout(fully_connected_layer, keep_prob) 
    return cropped 


@tfNamespace
def model_setup(x, keep_prob):
    """ Model setup """
    conv_layer1 = create_conv_layer(x, [5, 5, 1, 32], [32])
    conv_layer2 = create_subsequent_conv_layer(conv_layer1, [5, 5, 32, 64], [64]) 
    connected_layer1 = create_connected_layer(conv_layer2, 64)
    cropped_neurons1 = drop_neurons(connected_layer1, keep_prob) 
    logits = create_logits(cropped_neurons1) 
    #return {'keep_prob':keep_prob, 'logits':logits}
    return {'logits':logits}



###########################################
### Loss and activation functions below ###
###########################################

@tfNamespaceScalar
def cross_entropy(logits, labels):
    """ Execute the crossentropy activation function""" 
    return tf.reduce_mean(
                          tf.nn.softmax_cross_entropy_with_logits(
                          logits=logits,
                          labels=labels,
                          name='cross_entropy_loss'))

 
@tfNamespace
def train_optimizer(cross_entropy_loss, learning_rate):
    """ Train and optimize step"""
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    return optimizer.minimize(cross_entropy_loss)

    
@tfNamespaceScalar
def calc_accuracy(logits, labels):
    """ Return the mean accuracy"""
    correct_predictions = tf.equal(tf.argmax(logits, -1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))





