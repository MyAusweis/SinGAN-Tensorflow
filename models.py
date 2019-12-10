import tensorflow as tf
import numpy as np


"""
Generator

"""

def generator(x, y, config, name):
    x = conv_block(x, config["nfc"], config["ker_size"], name = name + "/generator/head")
    for i in range(config["num_layer"] - 2):
        N = int(config["nfc"]/(2**(i+1)))
        x = conv_block(x, max(N, config["min_nfc"]), config["ker_size"], name = name + "/generator/body/block_" + str(i+1))
    x = tf.layers.conv2d(x, config["nc_im"], kernel_size = config["ker_size"], activation = tf.nn.tanh, name = name + "/generator/tail", reuse = tf.AUTO_REUSE)
    y = tf.image.resize(y, size = (tf.shape(x)[1], tf.shape(x)[2]))
    return x + y


"""
Discriminator

"""

def discriminator(x, config, name):
    N = int(config["nfc"])
    x = conv_block(x, N, config["ker_size"], name = name + "/discriminator/head")
    for i in range(config["num_layer"] - 2):
        N = int(config["nfc"]/(2**(i+1)))
        x = conv_block(x, max(N, config["min_nfc"]), config["ker_size"], name = name + "/discriminator/body/block" + str(i+1))
    x = tf.layers.conv2d(x, 1, config["ker_size"], name = name + "/discriminator/tail", reuse = tf.AUTO_REUSE)
    return x


"""
Aux

"""

def conv_block(inputs, out_channels, ker_size, name):
    x = tf.layers.conv2d(inputs, out_channels, kernel_size = ker_size, activation = None, name = name + '/conv', reuse = tf.AUTO_REUSE)
    x = tf.layers.batch_normalization(x, name = name + '/bn', reuse = tf.AUTO_REUSE)
    x = tf.nn.leaky_relu(x, name = name + '/lr')
    return x
