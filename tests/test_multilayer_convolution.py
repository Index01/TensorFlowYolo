# ~*~ coding: utf-8 ~*~
#!/usr/bin/env python


import pytest
import tensorflow as tf
#from app import main_multilayer_convolutional as cuTest
from app import cnn_model_components as cuTest

print "\n[+] Engage Multilayer Convolution Tests, Starfighter.\n"

x = tf.placeholder(tf.float32, [100, 784])
conv_layer1 = cuTest.create_conv_layer(x, [5, 5, 1, 32], [32])   
conv_layer2 = cuTest.create_subsequent_conv_layer(conv_layer1, [5, 5, 32, 64], [64])
cropped_neurons, keep_prob = cuTest.drop_neurons(conv_layer2)
connected_layer = cuTest.create_connected_layer(cropped_neurons, 32)
cropped_neurons, keep_prob = cuTest.drop_neurons(connected_layer)
 
def test_mlcv_wshop():
    assert (cuTest.W_shop([5, 5, 1, 32])).shape==(5, 5, 1, 32)

def test_mlcv_bshop():
    assert (cuTest.b_shop([10])).shape==(10)

def test_mlcv_convolve():
    x_image = tf.reshape(x, [-1, 28, 28, 1]) 
    W = cuTest.W_shop([5, 5, 1, 32])
    b = cuTest.b_shop
    assert (cuTest.convolve(x_image, W)).shape==(100, 28, 28, 32)


def test_mlcv_pool():
    x_image = tf.reshape(x, [-1, 28, 28, 1]) 
    W_pool = cuTest.W_shop([5, 5, 1, 32])
    b_pool = cuTest.b_shop([32])
   
    # Some Setup 
    relu_conv = tf.nn.relu(cuTest.convolve(x_image, W_pool) + b_pool) 
    assert (cuTest.pool(relu_conv).shape)==(100, 14, 14, 32)


def test_mlcv_create_logits():
    connected_layer = cuTest.create_connected_layer(conv_layer2, 64)
    cropped_neurons, keep_prob = cuTest.drop_neurons(connected_layer)
    assert (cuTest.create_logits(cropped_neurons)).shape==(100, 10)


def test_mlcv_drop_neurons():
    connected_layer = cuTest.create_connected_layer(conv_layer2, 64)
    cropped_neurons, keep_prob = cuTest.drop_neurons(connected_layer)
    assert cropped_neurons.shape==(100, 1024)


def test_mlcv_create_convLayer():
    W_cl = [5, 5, 1, 32]
    b_cl = [32]
    assert (cuTest.create_conv_layer(x, W_cl, b_cl)).shape==(100, 14, 14, 32)

  
def test_mlcv_create_subsequent_conv_layer():
    W_cl2 = [5, 5, 32, 64]
    b_cl2 = [64]
    assert (cuTest.create_subsequent_conv_layer(conv_layer1, W_cl2, b_cl2)).shape==(100, 7, 7, 64) 


def test_mlcv_create_connectedLayer():
    
#    conv_layer = cuTest.create_conv_layer(x, [5, 5, 1, 32], [32])   
#    conv_layer2 = cuTest.create_subsequent_conv_layer(conv_layer, [5, 5, 32, 64], [64])
 
    assert (cuTest.create_connected_layer(conv_layer2, 64)).shape==(100, 1024)


def test_mlcv_modelSetup():
    print cuTest.model_setup(x).get('keep_prob')
    assert ((cuTest.model_setup(x)).get('keep_prob')).shape==()
    assert ((cuTest.model_setup(x)).get('logits')).shape==(100, 10)



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

