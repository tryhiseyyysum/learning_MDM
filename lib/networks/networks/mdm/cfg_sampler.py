#https://github.com/GuyTevet/motion-diffusion-model/blob/main/model/cfg_sampler.py

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy


class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  

        assert self.model.cond_mask_prob > 0, 'Cannot run, no conditions!'

        
        self.rot2xyz = self.model.rot2xyz
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.cond_mode = self.model.cond_mode
        self.encode_text = self.model.encode_text

    def forward(self, x, timesteps, y=None):
        cond_mode = self.model.cond_mode
        assert cond_mode in ['text']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out = self.model(x, timesteps, y)
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))

