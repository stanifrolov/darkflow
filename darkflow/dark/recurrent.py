from .layer import Layer

#TODO: implement recurrent layer

class recurrent_layer(Layer):
  def setup(self):
    self.num_units = 4102

  @property
  def signature(self):
    sig = ['recurrent']
    sig += self._signature[1:-2]
    return sig

  def finalize(self, _):
    """deal with darknet"""
    kernel = self.w['kernel']
    if kernel is None: return
    self.w['kernel'] = True