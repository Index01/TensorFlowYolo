# ~*~ coding: utf-8 ~*~
#! /usr/bin/env python



import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from dem_decorators import tfNamespace


"""
   Multilayer convolutional model.
"""


@tfNamespace
def W_shop(shape):
    """ Model weights"""
    W = tf.truncated_normal(shape, stddev=0.1, name='W')
    return tf.Variable(W)


@tfNamespace
def b_shop(shape):
    """ Model biases"""
    b = tf.constant(0.1, shape=shape, name='b')
    return tf.Variable(b)

@tfNamespace
def convolve(x, W):
    """ Convolve the image with the weight """
    strides = [1, 1, 1, 1]
    print x
    print W
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
        h_pool = pool(h_conv) 
        
        return h_pool


@tfNamespace
def create_subsequent_conv_layer(previous_layer, weights, biases):
    """ The subsequent layers need a slightly different method """

    with tf.name_scope("second_conv_layer"):
        W_conv2 = W_shop(weights)
        b_conv2 = W_shop(biases)
     
        h_conv2 = tf.nn.relu(convolve(previous_layer, W=W_conv2) + b_conv2) 
        h_pool2 = pool(h_conv2) 
        return h_pool2


@tfNamespace
def create_connected_layer(layer_in, dim1):
    """ Create a fully connected layer with relu"""
   
    with tf.name_scope("connected_layer"):
        flat_shape = [-1, 7 * 7 * dim1] 
        #flat_shape = [-1, 10 * 10 * dim1] 
        #print "layer_in: %s" % (layer_in)
        #print "flat_shape: %s" % (flat_shape)
  
        flattened = tf.reshape(layer_in, flat_shape) 
        W_flat = W_shop([(7 * 7 * dim1), 1024]) 
        #W_flat = W_shop([(10 * 10 * dim1), 1024]) 
        b_flat = b_shop([1024]) 

        return tf.nn.relu(tf.matmul(flattened, W_flat) + b_flat) 
        #return tf.nn.relu(dl_xent) 


@tfNamespace
def create_logits(drop_op):
    """ Accept a dense layer and return logits"""
    #return tf.layers.dense(inputs=dense_layer, units=10) 
    W_logs = W_shop([1024, 10]) 
    b_logs = b_shop([10]) 
    return tf.matmul(drop_op, W_logs) + b_logs
    #return tf.reshape(tf.matmul(drop_op, W_logs) + b_logs, [-1, 16, 16, 1])


@tfNamespace
def drop_neurons(fully_connected_layer):
    """ DDDDDdropout """
    keep_prob = tf.placeholder(tf.float32, shape=[])
    #print type(fully_connected_layer)
    cropped =tf.nn.dropout(fully_connected_layer, keep_prob) 
    return cropped, keep_prob


@tfNamespace
def model_setup(x):
    # Model setup
    conv_layer1 = create_conv_layer(x, [5, 5, 1, 32], [32])
    conv_layer2 = create_subsequent_conv_layer(conv_layer1, [5, 5, 32, 64], [64]) 
    connected_layer1 = create_connected_layer(conv_layer2, 64)
    cropped_neurons1, keep_prob = drop_neurons(connected_layer1) 
    logits = create_logits(cropped_neurons1) 
    return {'keep_prob':keep_prob, 'logits':logits}


@tfNamespace
def IO_setup():
    # Inputs and Outputs
    with tf.name_scope("input_output"):
        x = tf.placeholder(tf.float32, [None, 784], name='x')
        # Placeholder for the control
        y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
        return x, y_


def main():
    """Point of entry for our tensorflow mnist multilayer convolutional model.

       Currently only the training set is running, further paramaterization is
       needed and the test data set should be run at a standard interval. This
       stuff is really rough, we achieve ~91% accuracy with no dropout or batch
       or convolution ops. The purpose is to demo XEnt and Softmax with Tensorboard. 
    """
    print "[+] Welcome Starfighter"


    # Training parameters
    mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
    learning_rate = 0.01
    batch_size = 100
    #training_epochs = 1000
    training_epochs = 100
    train_logs = '../logs/train/multilayer_convolution'
    test_logs = '../logs/test/multilayer_convolution'
    train_writer = tf.summary.FileWriter(logdir=train_logs)
    test_writer = tf.summary.FileWriter(logdir=test_logs)
    x, y_ = IO_setup()
    model = model_setup(x),
    keep_prob = model[0].get('keep_prob') 
    logits=model[0].get('logits'), 
    
    # Loss functions
    with tf.name_scope("cross_entropy"):
        cross_entropy_loss = tf.reduce_mean(
                             tf.nn.softmax_cross_entropy_with_logits(
                             logits=logits, 
                             labels=y_,
                             name='cross_entropy_loss'))
   

    # Train
    with tf.name_scope("train_optimize"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(cross_entropy_loss)


    # Accuracy
    with tf.name_scope('correct_predictions'): 
        correct_predictions = tf.equal(tf.argmax(logits, -1),  tf.argmax(y_, 1)) 
 
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

 
    # Summary data for tensorboard
    tf.summary.scalar('cross_entropy', cross_entropy_loss)
    tf.summary.scalar('accuracy', accuracy)

    tf.summary.image('input', tf.reshape(x, [-1, 28, 28, 1]), 3)

#    tf.summary.histogram('weights', W)
#    tf.summary.histogram('biases', b)
    
    summary_merge = tf.summary.merge_all()



    # Init and Session
    init = tf.global_variables_initializer() 
    with tf.Session() as sess:
        sess.run(init) 
        for epoch in range(training_epochs):
            
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            #train.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})

            _, summary = sess.run([train, summary_merge],
                                  feed_dict={x:batch_x, 
                                             y_:batch_y, 
                                             keep_prob: 0.5})
 
            if epoch % 10 == 0:
                print "[+] Accuracy: %s" % (accuracy.eval(
                                            feed_dict={x:mnist.test.images, 
                                                       y_:mnist.test.labels,
                                                       keep_prob: 1.0}))

                #print "[+] keep_prob: %s" % (keep_prob) 

                # Generate the network graph for tensor board
                train_writer.add_summary(summary, epoch)
        train_writer.add_graph(sess.graph)


if __name__ == '__main__':
    main()