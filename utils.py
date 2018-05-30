
import cv2
import numpy as np
import tensorflow as tf

def resizeNscale(img, size):
    
    out = cv2.resize(img, (size, size)) # interpolation : bilinear
    
    _max = out.max()
    _min = out.min()
    
    out = (out - _min)/(_max - _min)
    
    return out
    
    
def extend(x):

    '''
    Extend tensor X
    b, h, w, c => b, 2h, 2w, c

    '''

    b,h, w, c = x.get_shape().as_list()



    out = tf.transpose(x, [0,3,1,2])       # (b, h, w, c) - > (b, c, h, w)
    out = tf.reshape(out, [-1,1])          # (b, c, h, w) - > (b*c*h*w, 1)
    out = tf.matmul(out, tf.ones([1,2]))      # (b*c*h*w, 1) - > (b*c*h*w, 2)


    out = tf.reshape(out, [-1, c, h, 2*w])  # (b*c*h*w, 2) - > (b, c, h, 2*w)
    out = tf.transpose(out, [0,1,3,2])     # (b, c, h, 2*w) - > (b, c, 2*w, h)
    out = tf.reshape(out, [-1,1])          # (b, c, 2*w, h) - > (b*c*2w*h, 1)
    out = tf.matmul(out, tf.ones([1,2]))      # (b*c*2w*h, 1) - > (b*c*2w*h, 2)


    out = tf.reshape(out, [-1, c, 2*w, 2*h]) # (b*c*2w*h, 2) - > (b, c, 2w, 2h)
    out = tf.transpose(out, [0, 3, 2, 1])

    return out



def extract_loc(x, y):

    '''
    x : input data at pooling layer
    y : extending pooling data
    x'shape == y'shape

    '''

    out = tf.equal(x, y)                  # tf.equal([[1,1],[3,4]], [[4,4],[4,4]]) = [[False, False],[False, True]]
    out = tf.cast(out, dtype=tf.float32)  # tf.cast([[False, False],[False, True]], dtype = tf.float32) = [[0.,0.],[0.,1.]]


    return out

def unpool2d(x, y):
    _x = extend(x)
    out = extract_loc(_x, y)
    return out

def init_w(name, shape):
    '''
    shape : [filter_h, filter_w, input_channel, ouput_channel]
    '''
    w = tf.get_variable(name, shape=shape,
                          initializer=tf.contrib.layers.xavier_initializer())
    return w

def init_b(name, shape):
    '''
    shape : [filter_h, filter_w, input_channel, ouput_channel]
    '''
    b = tf.get_variable(name, shape=shape,
                          initializer=tf.contrib.layers.xavier_initializer())
    return b

def max_pool_2d(_input, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME"):
    pool = tf.nn.max_pool(_input,  ksize=ksize, strides=strides, padding="SAME")
    return pool


def conv2d(_input, weight, strides = [1,1,1,1], padding = "SAME"):
    conv = tf.nn.conv2d(_input, weight, strides=strides, padding = padding)
    return conv



def deconv2d(_input, weight, _output_shape = None, strides = [1, 1, 1, 1], padding = "SAME"):
    _input_shape = _input.shape.as_list()
    weight_shape = weight.shape.as_list()

    if _output_shape == None:
        _output_shape = [tf.shape(_input)[0], _input_shape[1], _input_shape[2], weight_shape[3]]
    add_zero = tf.zeros([1, _output_shape[1], _output_shape[2], _output_shape[3],])
    deconv = tf.nn.conv2d_transpose(_input, weight, output_shape=_output_shape, strides=strides, padding=padding)
    return (deconv + add_zero)


def batch_norm(_input, center=True, scale=True, decay=0.8, is_training=True):
    norm = tf.contrib.layers.batch_norm(_input, center=center, scale = scale,
                                        decay = decay, is_training=is_training)
    return norm

def iou(logits, labels):

  
    x11, y11, x12, y12 = tf.split(logits, 4, axis=1)

    x21, y21, x22, y22 = tf.split(labels, 4, axis=1)



    xI1 = tf.maximum(x11, tf.transpose(x21))

    yI1 = tf.maximum(y11, tf.transpose(y21))



    xI2 = tf.minimum(x11+x12, tf.transpose(x21+x22))

    yI2 = tf.minimum(y11+y12, tf.transpose(y21+y22))



    inter_area = (xI2 - xI1) * (yI2 - yI1)



    logits_area = x12 * y12

    labels_area = x22 * y22

    union = (logits_area + tf.transpose(labels_area)) - inter_area

    inter_area = tf.cast(inter_area, tf.float32)
    union = tf.cast(union, tf.float32) + 1e-7



    return tf.maximum(inter_area / union, 0)
