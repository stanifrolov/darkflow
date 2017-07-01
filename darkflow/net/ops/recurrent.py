import numpy as np
import tensorflow as tf

from .baseop import BaseOp


# TODO: implement recurrent ops

class recurrent(BaseOp):
  def forward(self):
    num_units = self.lay.num_units
    _X = self.inp.out
    _X = tf.reshape(_X, [6, num_units]) #TODO: shape of _X (Tensor (?,13,13,30) and LSTM design not correct; compare with ROLO (FC layer 4096 neurons)
    _X = tf.split(_X, 6, 0)
    _istate = tf.zeros([1, num_units])
    cell = tf.contrib.rnn.LSTMCell(num_units)
    state = _istate

    for step in range(num_units):
      outputs, state = cell([_X[step]], state) #TODO: will the cell be zeroed after every forward pass or only first time init?
      tf.get_variable_scope().reuse_variables()

      self.out = outputs[0][:, 4097:4101]

  def speak(self):
    l = self.lay
    args = [l.ksize] * 2 + [l.pad] + [l.stride]
    args += [l.batch_norm * '+bnorm']
    args += [l.activation]
    msg = 'recurrent {}x{}p{}_{}  {}  {}'.format(*args)
    return msg
