# ~*~ coding: utf-8 ~*~
#! /usr/bin/env python



import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data




def main():
    """Point of entry for our tensorflow mnist training model using softmax and 
       cross entropy methods on the standard hand writing training set.

       Currently only the training set is running, further paramaterization is
       needed and the test data set should be run at a standard interval. This
       stuff is really rough, we achieve ~91% accuracy with no dropout or batch
       or convolution ops. The purpose is to demo XEnt and Softmax with Tensorboard. 
    """
    print "[+] Welcome Starfighter"


    # Training parameters
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    learning_rate = 0.01
    batch_size = 100
    training_epochs = 1000



    # Model Parameters
    with tf.name_scope("model_params"):
        W = tf.Variable(tf.zeros([784, 10]), name='W')
        b = tf.Variable(tf.zeros([10]), name='b')


    # Inputs and Outputs
    with tf.name_scope("input_output"):
        x = tf.placeholder(tf.float32, [None, 784], name='x')
        # Placeholder for the control
        y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

  
    # Define our model
    with tf.name_scope("model_definition"):
        cross_entropy_model = tf.matmul(x, W) + b

 
    # Loss functions
    with tf.name_scope("cross_entropy"):
        cross_entropy_loss = tf.reduce_mean(
                             tf.nn.softmax_cross_entropy_with_logits(
                             logits=cross_entropy_model, 
                             labels=y_,
                             name='cross_entropy_loss'))
   

    # Optimizer functions
    with tf.name_scope("train_optimize"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(cross_entropy_loss)


    # Accuracy
    with tf.name_scope('correct_predictions'): 
        correct_predictions = tf.equal(tf.argmax(cross_entropy_model, 1), 
                                                 tf.argmax(y_, 1)) 
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

 
    # Summary data for tensorboard
    tf.summary.scalar('cross_entropy', cross_entropy_loss)
    tf.summary.scalar('accuracy', accuracy)

    tf.summary.image('input', tf.reshape(x, [-1, 28, 28, 1]), 3)

    tf.summary.histogram('weights', W)
    tf.summary.histogram('biases', b)
    
    summary_merge = tf.summary.merge_all()



    # FileWriters for further introspection with tensorboard
    train_writer = tf.summary.FileWriter(logdir='./logs/train/softmax_crossentropy')
    test_writer = tf.summary.FileWriter(logdir='./logs/test/softmax_crossentropy')



    # Init and Session
    init = tf.global_variables_initializer() 
    with tf.Session() as sess:
        sess.run(init) 
         
        for epoch in range(training_epochs):
            
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, summary, iter_loss = sess.run([train, summary_merge, cross_entropy_loss], 
                                             feed_dict={x: batch_x, y_: batch_y})
            if epoch % 10 == 0:
                print "[+] Accuracy: %s" % (accuracy.eval(
                                            feed_dict={x:mnist.test.images, 
                                                       y_:mnist.test.labels}))
            
                # TODO: Review the loss returned by sess.run, something is up. 
                print "[+] Iterative_loss: %s" % (iter_loss) 
                train_writer.add_summary(summary, epoch)


        # Generate the network graph for tensor board
        train_writer.add_graph(sess.graph)
        
        # More debug stuff 
        curr_W, curr_b, curr_loss = sess.run([W, b, cross_entropy_loss], 
                                             feed_dict={x: batch_x, y_: batch_y})
        print "\n [+] Final values:" 
        print("W: %s ,\n b: %s,\n loss: %s" % (curr_W, curr_b, curr_loss))




if __name__ == '__main__':
    main()
