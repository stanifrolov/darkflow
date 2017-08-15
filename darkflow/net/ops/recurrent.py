import tensorflow as tf

from .baseop import BaseOp


class recurrent(BaseOp):
  def forward(self):
    _X = self.inp.out
    input_shape = tf.shape(_X)[0]
    num_units = _X.shape.dims[1].value * _X.shape.dims[2].value * _X.shape.dims[3].value
    _X_list = tf.reshape(_X, [input_shape, num_units])
    _X_list = tf.split(_X_list, self.lay.seq_length, 0)
    batch_size = tf.shape(_X_list[0])[0]

    with tf.variable_scope(self.scope) as scope:
      cell = tf.contrib.rnn.LSTMCell(num_units, state_is_tuple=True)
      state = (tf.zeros([batch_size, num_units]),) * 2
      #state = tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size, num_units]), tf.zeros([batch_size, num_units]))
      out = []

      for step in range(self.lay.seq_length):
        if step > 0:
          scope.reuse_variables()
        outputs, state = cell(_X_list[step], state)
        out.append(outputs)

    out = tf.stack(out, 0)
    self.out = tf.reshape(out, [input_shape, _X.shape.dims[1].value, _X.shape.dims[2].value, _X.shape.dims[3].value])

  def speak(self):
    l = self.lay
    # args = [l.ksize] * 2 + [l.pad] + [l.stride]
    # args += [l.batch_norm * '+bnorm']
    # args += [l.activation]
    args = [l.seq_length]
    msg = 'recurrent seq_length {}'.format(*args)
    return msg
