# ~*~ coding: utf-8 ~*~
#! /usr/bin/env python


import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import cnn_model_components as cnn
from log_utils import Projector, TBLogger

"""
   Multilayer CNN model.
"""

def IO_setup():
    """ Inputs and Outputs """
    with tf.name_scope("input_output"):
        x = tf.placeholder(tf.float32, [None, 784], name='x')
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
    #training_epochs = 100
    training_epochs = 1000
#    train_logs = os.path.abspath('../logs/train/multilayer_convolution')
#    test_logs = os.path.abspath('../logs/test/multilayer_convolution')
#    train_writer = tf.summary.FileWriter(logdir=train_logs)
#    test_writer = tf.summary.FileWriter(logdir=test_logs)
#

    # Model
    x, y_ = IO_setup()
    model = cnn.model_setup(x)
    print 'model: %s' % (model)
    keep_prob = model.get('keep_prob') 
    logits=model.get('logits') 
   
 
    # Logs
    tbLogger = TBLogger(logdir='multilayer_convolution')
    tbLogger.image_summary(x)
    logProjector = Projector(log_dir=tbLogger.get_train_log_dir(), 
                             metadata_labels=input_data.read_data_sets("../MNIST_data").test.labels, 
                             metadata_data=mnist.test.images) 



    # Loss and Train
    cross_entropy_loss = cnn.cross_entropy(logits=logits, labels=y_)
    train = cnn.train_optimizer(cross_entropy_loss, learning_rate) 
    accuracy = cnn.calc_accuracy(logits, y_)

    summary_merge = tbLogger.merge_summaries() 



#    # Loss functions
#    with tf.name_scope("cross_entropy"):
#        cross_entropy_loss = tf.reduce_mean(
#                                            tf.nn.softmax_cross_entropy_with_logits(
#                                            logits=logits, 
#                                            labels=y_,
#                                            name='cross_entropy_loss'))
#   
#
#    # Train
#    with tf.name_scope("train_optimize"):
#        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#        train = optimizer.minimize(cross_entropy_loss)
#
#
#    # Accuracy
#    with tf.name_scope('correct_predictions'): 
#        correct_predictions = tf.equal(tf.argmax(logits, -1),  tf.argmax(y_, 1)) 
#    with tf.name_scope('accuracy'):
#        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
#
#
#    # Summary data for tensorboard
#    tf.summary.scalar('cross_entropy', cross_entropy_loss)
#    tf.summary.scalar('accuracy', accuracy)
#    tf.summary.image('input', tf.reshape(x, [-1, 28, 28, 1]), 3)
#
#    summary_merge = tf.summary.merge_all()
#

    # Init and Session
    init = tf.global_variables_initializer() 
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            #train.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
            _, summary = sess.run([train, summary_merge],
                                  feed_dict={x:batch_x, 
                                             y_:batch_y, 
                                             keep_prob: 0.5})
            if epoch % 100 == 0:
#"""
#                print "[+] Accuracy: %s" % (accuracy.eval(
#                                            feed_dict={x:mnist.test.images, 
#                                                       y_:mnist.test.labels,
#                                                       keep_prob: 1.0}))
#"""
#                print "[+] keep_prob: %s" % (keep_prob) 

                logProjector.update(sess, epoch)
                tbLogger.write_train_state(summary, epoch)
                #train_writer.add_summary(summary, epoch)
        tbLogger.write_train_graph(sess)
        #train_writer.add_graph(sess.graph)

if __name__ == '__main__':
    main()
