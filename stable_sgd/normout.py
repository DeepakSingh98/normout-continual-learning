from os import truncate
import torch
import torch.nn as nn


class NormOut(nn.Module):
    """
    Sets ith neurons to zero with probability p_i, where p_i is the activation of the ith neuron divided 
    by the max activation of the layer. When `use_abs` is True, we use the absolute value of the activations 
    instead of the activations themselves.
    """
    def __init__(self, normalization_type: str, use_abs=True, **kwargs):

        nn.Module.__init__(self)

        self.use_abs = use_abs
        self.normalization_type = normalization_type

    def forward(self, x):

        if self.training:

            if self.use_abs:
                x_prime = abs(x)
            else:
                x_prime = x
            
            if self.normalization_type == "TemporalMax":
                # Take max across batch
                x_prime_max = torch.max(x_prime, dim=0, keepdim=True)[0]
                norm_x = x_prime / x_prime_max

            elif self.normalization_type == "SpatialMax":
                # Take max across layer
                x_prime_max =  torch.max(x_prime, dim=1, keepdim=True)[0]
                norm_x = x_prime / x_prime_max

            elif self.normalization_type == "SpatiotemporalMax":
                # Take max across batch
                x_prime_max = torch.max(x_prime, dim=0, keepdim=True)[0]
                # Take max across layer
                x_prime_max = torch.max(x_prime_max, dim=1, keepdim=True)[0]
                norm_x = x_prime / x_prime_max

            else:
                raise NotImplementedError("normalization type not implemented")
 
            x_mask = torch.rand_like(x) < norm_x
            x = x * x_mask

        return x
