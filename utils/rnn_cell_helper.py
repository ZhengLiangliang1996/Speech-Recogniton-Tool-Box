import tensorflow as tf

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