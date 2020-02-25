import numpy as np
import tensorflow as tf
import sys
sys.path.append('/home/liangliang/Desktop/VUB_ThirdSemester/MasterThesis/SpeechRecognitiontoolbox')

from utils.spectral_norm import spectral_normed_weight


def conv1d(inputs, filters, kernel_size, stride = 1, dilation = 1, activation = 'relu', name="conv1d", gn = '1'):
    with tf.compat.v1.variable_scope(name+gn):
        w = tf.compat.v1.get_variable(name+"w",
                           shape=(kernel_size, inputs.get_shape().as_list()[-1], filters), 
                           dtype = tf.float32, 
                           initializer = tf.initializers.orthogonal(),
                           )
        #SN
        w = spectral_normed_weight(w)
        outputs = tf.nn.conv1d(input = inputs, filters = w, padding = 'SAME', stride = stride, dilations = dilation)
        if activation == 'relu':
            outputs = tf.nn.relu(outputs)
        if activation =='tanh':
            outputs = tf.nn.tanh(outputs)
        return outputs


#需要修改 shape 之间的关系
def conv_transpose(inputs, output_shape, strides, name="convtrans1d", gn='1'):
    """
    inputs is 
    filters is outputs
    strides is upsample_factor
    """

    with tf.compat.v1.variable_scope(name+gn):
        w = tf.compat.v1.get_variable(name+"w",
                            # [kernel_width, output_depth, input_depth]
                           shape=(strides * 2, output_shape * strides, inputs.get_shape().as_list()[-1]), 
                           dtype = tf.float32, 
                           initializer = tf.initializers.orthogonal(),
                           )
        w = spectral_normed_weight(w)
        outputs = tf.nn.conv1d_transpose(input = inputs, filters = w, output_shape = [inputs.get_shape().as_list()[0], inputs.get_shape().as_list()[1], output_shape *  strides], strides = strides, padding = "SAME", data_format="NWC")

    return outputs  


def conditionalBatchnorm(inputs, noise, output=128, name="cbn", gn='1'):
    with tf.compat.v1.variable_scope(name+gn):
        output1 = tf.compat.v1.layers.batch_normalization(inputs = inputs, training=True)

        num_features = inputs.get_shape().as_list()[-1]
        w = tf.compat.v1.get_variable(name+"w", 
            shape=(output, num_features*2), 
            dtype=tf.float32,
            initializer= tf.initializers.orthogonal(),
            )

        w = spectral_normed_weight(w)

        mul = tf.matmul(noise, w)

        bias = tf.compat.v1.get_variable(name+"b", [num_features * 2], initializer=tf.zeros_initializer())
        mul = tf.nn.bias_add(mul, bias)

        gamma, beta = tf.split(value = mul, num_or_size_splits=2, axis = 1)
        print(gamma.get_shape().as_list())
        gamma = tf.reshape(gamma, [-1, tf.shape(inputs)[-1]])
        beta = tf.reshape(beta, [-1, tf.shape(inputs)[-1]])

        outputs = gamma * output1 + beta
    
    return outputs

def gblock(inputs, z, hidden_channel, upsample_factor, gblock_name = '1'):
    # First Stack
    outputs = conditionalBatchnorm(inputs, z, name = 'cbn1', gn = gblock_name)
    outputs = conv_transpose(outputs, outputs.get_shape()[-1], upsample_factor, name='convtranspose1', gn = gblock_name)
    outputs = conv1d(outputs, hidden_channel, kernel_size = 3, name='conv1d1', gn = gblock_name)

    # Second Stack
    outputs = conditionalBatchnorm(outputs, z, name='cbn2', gn = gblock_name)
    outputs = conv_transpose(outputs, outputs.get_shape()[-1], upsample_factor, name='convtranspose2', gn = gblock_name)
    outputs = conv1d(outputs, hidden_channel, kernel_size = 3, dilation = 2, name='conv1d2', gn = gblock_name)

    # Residule Part 
    residual_outputs = conv_transpose(inputs, inputs.get_shape()[-1], upsample_factor, name='residual1', gn = gblock_name)
    residual_outputs = conv1d(residual_outputs, hidden_channel, kernel_size = 1, name='resudualconv1d1', gn = gblock_name)

    # Third Stack
    outputs = conditionalBatchnorm(residual_outputs, z, name='cbn3', gn = gblock_name)
    outputs = conv1d(outputs, hidden_channel, kernel_size = 3, dilation = 4, name='conv1d3', gn = gblock_name)

    # Fourth Stack
    outputs = conditionalBatchnorm(outputs, z, name='cbn4', gn = gblock_name)
    outputs = conv1d(outputs, hidden_channel, kernel_size = 3, dilation = 8, name='conv1d4', gn = gblock_name)

    outputs = outputs + residual_outputs
    return outputs

def generator(x, z):
    input_pre = conv1d(inputs = x, filters = 768, kernel_size = 3, stride = 1, dilation = 1, activation = 'relu', name="conv1d")
    g1 = gblock(input_pre, z, 768, 1, gblock_name = '1')
    g2 = gblock(g1, z, 768, 1, gblock_name = '2')
    g3 = gblock(g2, z, 384, 2, gblock_name = '3')
    g4 = gblock(g3, z, 384, 2, gblock_name = '4')
    g5 = gblock(g4, z, 384, 2, gblock_name = '5')
    g6 = gblock(g5, z, 192, 3, gblock_name = '6')
    g7 = gblock(g6, z, 96, 5, gblock_name = '7')
    outputs = conv1d(inputs = g7, filters = 1, kernel_size =3, activation = 'tanh', name='conv1d8')
    return outputs



# 模型测试
#(567, 128)
# print(output)

# g = tf.Graph()
# with g.as_default():
#     # data shape is "[batch, in_height, in_width, in_channels]",
    
#     x = tf.Variable(tf.random.normal([80, 1, 1293], stddev=0.35),name="weights")
#     # tf.print(tf.shape(x))
#     z = tf.Variable(tf.random.normal([80, 128], stddev=0.35),name="weights")
#     # tf.print(tf.shape(z))
#     # filter shape is "[filter_height, filter_width, in_channels, out_channels]"
#     # filters = tf.Variable(, name="phi")
#     outputs = generator(x, z)
#     print(outputs)
    
    

