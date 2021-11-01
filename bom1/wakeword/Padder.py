#Pads the audio to a certain length.
import torch

class Padder:
  '''
  Pads to a desired length.
  '''
  def __init__(self, outlen):
    self.outlen = outlen
    return
  
  def __call__(self, x, y = None):
    #x is audio, y is sample rate
    if x.shape[-1] > self.outlen:
      x = x[:, :self.outlen]
    assert x.shape[-1] <= self.outlen, f'outlen should be larger than the length of x. outlen: {self.outlen}, x.shape: {x.shape}'
    x = torch.nn.ConstantPad1d((0, self.outlen - x.shape[-1]), 0)(x)

    assert x.shape[-1] == self.outlen, f'Something went wrong.'
    return x, y