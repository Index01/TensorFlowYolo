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

def test_mlcv_modelSetup():
    x = tf.placeholder(tf.float32, [100, 784]) 
    assert ((cuTest.model_setup(x)).get('keep_prob')).shape==()
    #assert ((cuTest.model_setup(x)).get('logits')).shape==()
    assert ((cuTest.model_setup(x)).get('logits'))[0].shape==(100, 10)

def test_mlcv_convolve():
    x = 
    W = 
    print cuTest.convolve(x, W)


def test_mlcv_pool():
    pass

def test_mlcv_create_convLayer():
    pass

def test_mlcv_create_connectedLayer():
    pass

def test_mlcv_create_logits():
    pass

def test_mlcv_drop_neurons():
    pass




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

