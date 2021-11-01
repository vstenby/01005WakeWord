import torch

class TransformMono:
    '''
    Transforms a clip to mono.
    '''
    def __call__(self, x, y=None):
        x = torch.mean(x, dim=0).unsqueeze(0)
        return x, y