from .layer import Layer

class recurrent_layer(Layer):
  def setup(self, seq_length):
    self.seq_length = seq_length

  @property
  def signature(self):
    sig = ['recurrent']
    sig += self._signature[1:-2]
    return sig

  def finalize(self, _):
    """deal with darknet"""
    kernel = self.w['kernel']
    if kernel is None:
      return
    self.w['kernel'] = True