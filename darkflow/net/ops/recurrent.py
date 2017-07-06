import tensorflow as tf

from .baseop import BaseOp


class recurrent(BaseOp):
  def forward(self):
    _X = self.inp.out
    input_shape = tf.shape(_X)[0]
    num_units = _X.shape.dims[1].value * _X.shape.dims[2].value * _X.shape.dims[3].value
    _X_list = tf.reshape(_X, [input_shape, num_units])  # TODO: get shape dynamically from last layer
    _X_list = tf.split(_X_list, 1, 0)

    cell = tf.contrib.rnn.LSTMCell(num_units, state_is_tuple=False)
    state = tf.zeros([input_shape, 2 * num_units])

    with tf.variable_scope(tf.get_variable_scope()) as _:
      for step in range(3):
        outputs, state = cell(_X_list[0], state)
        tf.get_variable_scope().reuse_variables()

    self.out = tf.reshape(outputs, [input_shape, _X.shape.dims[1].value, _X.shape.dims[2].value, _X.shape.dims[3].value])

  def speak(self):
    l = self.lay
    # args = [l.ksize] * 2 + [l.pad] + [l.stride]
    # args += [l.batch_norm * '+bnorm']
    # args += [l.activation]
    args = [l.seq_length]
    msg = 'recurrent seq_length {}'.format(*args)
    return msg
