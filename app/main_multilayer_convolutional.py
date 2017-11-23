# ~*~ coding: utf-8 ~*~
#! /usr/bin/env python

"""
   Multilayer CNN model for solving the 10,000 image Mnist set.
"""

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import cnn_model_components as cnn
from log_utils import Projector, TBLogger


def main():
    """Point of entry for our tensorflow mnist multilayer convolutional model.

       This model runs with components from the cnn module and reduces loss through
       the cross entropy functions from Tensorflow. The model currently achieves approx 98% accuracy
       in recognizing the 10,000 handwritten digits 0-9 from the MNIST training set, with no 
       particular pre-processing or sorting.
    """
    print "[+] Welcome Starfighter"


    # Training parameters
    mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
    mnist_log_ds = input_data.read_data_sets("../MNIST_data", one_hot=False)
    learning_rate = 0.01
    batch_size = 50
    training_epochs = 2000


    # Model
    x, y_ = cnn.IO_setup()
    keep_prob = tf.placeholder(tf.float32) 
    model = cnn.model_setup(x, keep_prob)
    logits = model.get('logits') 
 
    # Logs
    tbLogger = TBLogger(logdir='multilayer_convolution')
    tbLogger.image_summary(x)
    logProjector = Projector(log_dir=tbLogger.get_train_log_dir(), 
                             metadata_labels=mnist_log_ds.train.labels, 
                             metadata_data=mnist_log_ds.train.images) 


    # Loss and Train
    cross_entropy_loss = cnn.cross_entropy(logits=logits, labels=y_)
    train = cnn.train_optimizer(cross_entropy_loss, learning_rate) 
    accuracy = cnn.calc_accuracy(logits, y_)

    summary_merge = tbLogger.merge_summaries() 


    # Init and Session
    init = tf.global_variables_initializer() 
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            test_x, test_y = mnist.test.next_batch(batch_size)
            _, summary = sess.run([train, summary_merge],
                                  feed_dict={x:batch_x, 
                                             y_:batch_y, 
                                             keep_prob: 1.0})

            if epoch % 100 == 0:
                print epoch
                acc, summary = sess.run([accuracy, summary_merge],
                                         feed_dict={x:test_x, 
                                                    y_:test_y, 
                                                    keep_prob: 1.0})
                print "[+]Accuracy: %s" % (acc)

                tbLogger.write_test_state(summary, epoch)
                logProjector.update(sess, epoch)

            tbLogger.write_train_state(summary, epoch)

        tbLogger.write_train_graph(sess)

if __name__ == '__main__':
    main()
