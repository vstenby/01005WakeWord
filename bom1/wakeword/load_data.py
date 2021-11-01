import torch
import torchaudio
import torchaudio.transforms as T

def load_data(path, f, sr = 22050, normalize = False, transforms = None):
    '''
    Load in audio, sample-rate and x (either MFCC or spectrogram).
    '''
    
    audio, sr0 = torchaudio.load(path, normalize = normalize)
    
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