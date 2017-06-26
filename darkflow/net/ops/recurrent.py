import tensorflow as tf

from .baseop import BaseOp


# TODO: implement recurrent ops

class recurrent(BaseOp):
  def forward(self):
    cell = tf.contrib.rnn.LSTMCell(5002)
    pass

  def speak(self):
    l = self.lay
    args = [l.ksize] * 2 + [l.pad] + [l.stride]
    args += [l.batch_norm * '+bnorm']
    args += [l.activation]
    msg = 'recurrent {}x{}p{}_{}  {}  {}'.format(*args)
    return msg
