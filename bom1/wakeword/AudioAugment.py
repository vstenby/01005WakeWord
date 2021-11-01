import numpy as np
import augment
import torch

class AudioAugment:
  '''
  AudioAugmentor
  '''
  def __init__(self, reverb, snr, pitch, p):
    #Set the parameters
    self.reverb = reverb
    self.snr    = snr
    self.pitch  = pitch
    self.p      = p

  def __call__(self, x, sr):
    #Draw reverb, snr and pitch.

    flips = np.random.binomial(1, p=self.p, size=3)

    if flips[0]:
      #Add reverb
      reverb = np.random.randint(0, self.reverb)
      x = augment.EffectChain().reverb(reverb, reverb, reverb).channels(2).apply(x, src_info={'rate': sr})

    if flips[1]:
      #Add noise
      noise_generator = lambda: torch.zeros_like(x).uniform_()
      x = augment.EffectChain().additive_noise(noise_generator, snr=self.snr).apply(x, src_info={'rate': sr})

    if flips[2]:
      #Add pitch - PITCH CAN SOMEHOW CHANGE THE SHAPE. NOT SURE HOW, WHY ETC.
      pitch = np.random.randint(-self.pitch, self.pitch)
      x = augment.EffectChain().pitch(pitch).rate(sr).apply(x, src_info = {'rate' : sr})
      
    return x, sr