# ~*~ coding: utf-8 ~*~
#!/usr/bin/env python


import pytest
import tensorflow as tf
from app import main_multilayer_convolutional as cuTest

print "\n[+] Engage Multilayer Convolution Tests, Starfighter.\n"

def test_mlcv_wshop():
    assert (cuTest.W_shop([5, 5, 1, 32])).shape==(5, 5, 1, 32)

def test_mlcv_bshop():
    assert (cuTest.b_shop([10])).shape==(10)

def test_mlcv_convolve():
    x = tf.placeholder(tf.float32, [100, 784]) 
    x_image = tf.reshape(x, [-1, 28, 28, 1]) 
    W = cuTest.W_shop([5, 5, 1, 32])
    b = cuTest.b_shop
    assert (cuTest.convolve(x_image, W)).shape==(100, 28, 28, 32)


def test_mlcv_pool():
    x = tf.placeholder(tf.float32, [100, 784]) 
    x_image = tf.reshape(x, [-1, 28, 28, 1]) 
    W_pool = cuTest.W_shop([5, 5, 1, 32])
    b_pool = cuTest.b_shop([32])
   
    # Some Setup 
    relu_conv = tf.nn.relu(cuTest.convolve(x_image, W_pool) + b_pool) 
    assert (cuTest.pool(relu_conv).shape)==(100, 14, 14, 32)


def test_mlcv_create_logits():
    x = tf.placeholder(tf.float32, [100, 784]) 
    conv_layer = cuTest.create_convolutional_layer(x, [5, 5, 1, 32], [32])   
    conv_layer2 = cuTest.create_second_conv_layer(conv_layer, [5, 5, 32, 64], [64])
    connected_layer = cuTest.create_connected_layer(conv_layer2)
    cropped_neurons, keep_prob = cuTest.drop_neurons(connected_layer)
    assert (cuTest.create_logits(cropped_neurons)).shape==(100, 10)


def test_mlcv_drop_neurons():
     
    x = tf.placeholder(tf.float32, [100, 784]) 
    conv_layer = cuTest.create_convolutional_layer(x, [5, 5, 1, 32], [32])
    conv_layer2 = cuTest.create_second_conv_layer(conv_layer, [5, 5, 32, 64], [64])
    connected_layer = cuTest.create_connected_layer(conv_layer2)
    cropped_neurons, keep_prob = cuTest.drop_neurons(connected_layer)
    assert cropped_neurons.shape==(100, 1024)


def test_mlcv_create_convLayer():
    x = tf.placeholder(tf.float32, [100, 784]) 
    W_cl = [5, 5, 1, 32]
    b_cl = [32]
    assert (cuTest.create_convolutional_layer(x, W_cl, b_cl)).shape==(100, 14, 14, 32)

  
def test_mlcv_create_second_conv_layer():
    x = tf.placeholder(tf.float32, [100, 784]) 
    W_cl1 = [5, 5, 1, 32]
    W_cl2 = [5, 5, 32, 64]
    b_cl1 = [32]
    b_cl2 = [64]
    prev_layer = cuTest.create_convolutional_layer(x, W_cl1, b_cl1)
    assert (cuTest.create_second_conv_layer(prev_layer, W_cl2, b_cl2)).shape==(100, 7, 7, 64) 


def test_mlcv_create_connectedLayer():
    x = tf.placeholder(tf.float32, [100, 784]) 
    conv_layer = cuTest.create_convolutional_layer(x, [5, 5, 1, 32], [32])  
    conv_layer2 = cuTest.create_second_conv_layer(conv_layer, [5, 5, 32, 64], [64])
    assert (cuTest.create_connected_layer(conv_layer2)).shape==(100, 1024)


def test_mlcv_modelSetup():
    x = tf.placeholder(tf.float32, [100, 784]) 
    assert ((cuTest.model_setup(x)).get('keep_prob')).shape==()
    assert ((cuTest.model_setup(x)).get('logits'))[0].shape==(100, 10)



#class TestMultilayerConvolution(tf.test.TestCase):
#    """ Creating a test object which inherits from tensorflow, ultimately unittet"""


#    def test_setup():
#        """ Setup or fixtures """
#        print "[+] Setup"


#    def test_mlconv():
#        """ Test cases """
#        print "[+] Test ML Convo"
 

#    def test_teardown():
#        """ Teardown or fixtures """
#        print "[+] Teardown"




def test_main():
    """" Main unit tests for multilayer convolution tensorflow modules."""

"""
    MyMLConTest = TestMultilayerConvolution() 
    MyMLConTest.test_setup()
    MyMLConTest.test_mlconv()
    MyMLConTest.test_teardown()
"""


if __name__ == '__test_main__':
    test_main()

