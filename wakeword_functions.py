import numpy  as np
import pandas as pd
import subprocess
import os 

import torch
import torchaudio
import torchaudio.transforms as T
import augment
import multiprocessing

import tqdm.auto as tqdm

def get_mask(ID, t, n, label = False):
    '''
    Gets a mask where the keywords are. This is similar to get_targets.
    '''
    
    assert len(t.shape) == 2, 't should have [n x 2] shape.'
    assert t.shape[-1]  == 2, 't should have [n x 2] shape.'
    
    #First, we want to get the duration.
    duration   = get_duration(ID)
    
    t_linspace = np.linspace(0, duration, n)
    targets    = np.zeros_like(t_linspace)
    
    t1s = t[:,0]
    t2s = t[:,1]
    
    mask_value = 1
    for t1, t2 in zip(t1s, t2s):
        targets[(t1 <= t_linspace)&(t_linspace <= t2)] = mask_value
        if label: mask_value += 1
        
    return t_linspace, targets

def get_targets(ID, t, n, delay = 0, target_duration = 1):
    '''
    Returns the targets of a lecture. 
    
    Inputs:
        ID              : ID to the lecture
        t               : array of size n x 2, where t[:,0] is t1 (start of wakeword) and t[:,1] is t2 (end of wakeword)
        n               : desired output length
        delay           : delay from t2 until target in seconds.
        target_duration : how long target should be 1 after t2+delay.
    '''
    
    assert len(t.shape) == 2, 't should have [n x 2] shape.'
    assert t.shape[-1]  == 2, 't should have [n x 2] shape.'
    
    #First, we want to get the duration.
    duration   = get_duration(ID)
    
    t_linspace = np.linspace(0, duration, n)
    targets    = np.zeros_like(t_linspace)
    
    #Fetch end of keyword utterances and add delay.
    t2s = t[:,1] + delay
    
    for t2 in t2s:
        timediffs = t_linspace - t2
        targets[(timediffs > 0)&(timediffs <= target_duration)] = 1
    
    return t_linspace, targets

# --- Data related functions ---

def load_data(path, f, sr = 22000, normalize = True, transforms = None):
    '''
    Load in audio, sample-rate and x (either MFCC or spectrogram).
    '''
    
    audio, sr0 = torchaudio.load(path, normalize = normalize)
    
    if normalize:
        audio, sr0 = torchaudio.sox_effects.apply_effects_tensor(audio, sr0, [['gain', '-n']], channels_first=True)
    
    #Resample to the desired sample rate.
    audio = T.Resample(sr0, sr)(audio)
    
    if transforms:
        if type(transforms) is not list:
            transforms = [transforms]
        
        for t in transforms:
            audio, sr = t(audio, sr)
            
    x = f(audio)
    return audio, sr, x

# --- Transformers ---

#Pads the audio to a certain length.
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


class TransformMono:
    '''
    Transforms a clip to mono.
    '''
    def __call__(self, x, y=None):
        x = torch.mean(x, dim=0).unsqueeze(0)
        return x, y
    
    
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

class SoxTransform:
  '''
  A wrapper for torchaudio.sox_effects.apply_effects_tensor.
  '''
  def __init__(self, effects):
    self.effects = effects

  def __call__(self, x, y):
    x, y = torchaudio.sox_effects.apply_effects_tensor(x, y, self.effects, channels_first=True)
    return x, y
