import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np

def load_data(path, f, sr = 22050, normalize = False, transforms = None):
    '''
    Load in audio, sample-rate and x (either MFCC or spectrogram).
    '''
    
    #if t1 is None and t2 is None:
        #Load the entire thing.
    audio, sr0 = torchaudio.load(path, normalize = normalize)
    #else:
        #Load from t1 to t2.
    #    sr0 = torchaudio.info(path).sample_rate
    #    frame_offset = int(np.round(t1 * sr0))
    #    num_frames = int(np.round((t2-t1)*sr0))
    #    audio, sr0 = torchaudio.load(path, normalize = normalize, frame_offset = frame_offset, num_frames = num_frames)
        
    #    assert audio.shape[-1] != 0, f'audio.shape[-1] found to have size 0.'
        
    #if normalize:
    #    audio, sr0 = torchaudio.sox_effects.apply_effects_tensor(audio, sr0, [['gain', '-n']], channels_first=True)
    
    #Resample to the desired sample rate.
    
    if sr0 != sr:
        audio = T.Resample(sr0, sr)(audio)
    
    if transforms:
        if type(transforms) is not list:
            transforms = [transforms]
        
        for t in transforms:
            audio, sr = t(audio, sr)
            
    x = f(audio)
    return audio, sr, x