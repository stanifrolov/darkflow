import numpy as np
import tensorflow as tf

from .baseop import BaseOp


# TODO: implement recurrent ops

class recurrent(BaseOp):
  def forward(self):
    _X = self.inp.out
    input_shape = tf.shape(_X)[0]
    _X = tf.reshape(_X, [input_shape, 19*19*30]) # TODO: get shape from last layer
    _X = tf.split(_X, 1, 0)

    cell = tf.contrib.rnn.LSTMCell(13*13*30, state_is_tuple=False)
    state = tf.zeros([input_shape, 2 * 13*13*30])

    with tf.variable_scope(tf.get_variable_scope()) as _:
      for step in range(3):
        outputs, state = cell(_X[0], state)
        tf.get_variable_scope().reuse_variables()

    self.out = tf.reshape(outputs, [input_shape, 13, 13, 30])

  def speak(self):
    #l = self.lay
    #args = [l.ksize] * 2 + [l.pad] + [l.stride]
    #args += [l.batch_norm * '+bnorm']
    #args += [l.activation]
    #msg = 'recurrent {}x{}p{}_{}  {}  {}'.format(*args)
    #return msg
    return "recurrent"
