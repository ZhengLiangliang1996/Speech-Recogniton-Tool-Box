import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras.layers.legacy_rnn.rnn_cell_impl import LayerRNNCell

class RNN_cell(object):
    #@classmethod#工厂模式
    @classmethod
    def make_cell(cls, cell_type, layer_norm):
        if layer_norm == False:
            if cell_type == 'gru':
                return tf.contrib.rnn.GRUCell
            elif cell_type == 'lstm':
                return tf.contrib.rnn.BasicLSTMCell
            elif cell_type == 'rnn':
                return tf.contrib.rnn.RNNCell
            raise Exception("rnn celltype input wrongly, only support rnn, gru, lstm")
        else:
            if  cell_type == 'gru':
                #return tf.contrib.rnn.LayerNormGRUCell
                raise Exception('Not implemented yet, try rnn without layer norm')
            elif cell_type == 'lstm':
                return tf.contrib.rnn.LayerNormBasicLSTMCell
            elif cell_type == 'rnn':
                raise Exception('Not implemented yet, try rnn without layer norm')
            raise Exception("rnn celltype input wrongly, only support rnn, gru, lstm")


class ModifiedLSTM(LayerRNNCell):
    def __init__(self,
                 units,
                 activation,
                 recurrent_min=0,
                 recurrent_max=None,
                 recurrent_kernel_initializer=None,
                 input_kernel_initializer=None,
                 reuse=0,
                 name=None)
        super(LayerRNNCell, self)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._units = units
    self._recurrent_min = recurrent_min
    self._recurrent_max = recurrent_max
    self._recurrent_initializer = recurrent_kernel_initializer
    self._input_initializer = input_kernel_initializer
    self._activation = activation or nn_ops.relu

 	@property
 	def state_size(self):
 		return self._units

	@property
	def output_size(self):
		return self._units

	def build(self, input_shape):

