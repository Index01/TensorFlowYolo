# ~*~ coding: utf-8 ~*~
#! /usr/bin/env python



import tensorflow as tf




def main():
    """Point of entry for tensor flow test project"""
    print "[+] Welcome Starfighter"



    # Model Parameters
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)


    # Inputs and Outputs
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    linear_model = W * x + b

    # Loss functions
    loss = tf.reduce_sum(tf.square(linear_model - y))


    # Optimizer functions
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss) 


    # Training data
    x_train = [1, 2, 3, 4] 
    y_train = [0, -1, -2, -3] 






    # Init and Session
    init = tf.global_variables_initializer() 
    with tf.Session() as sess:
        sess.run(init) 
        for i in range(1000):
            sess.run(train, {x: x_train, y: y_train})

        # Evaluate the training accuracy
        curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
        print("W: %s, b: %s, loss: %s" % (curr_W, curr_b, curr_loss))






if __name__ == '__main__':
    main()
