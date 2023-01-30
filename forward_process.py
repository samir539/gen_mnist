import torch
import torch.nn.functional as F
import matplotlib as plt

#we need to 
# read in signatures
# apply noise

def beta_vals(timesteps,start=0.0001,stop=0.02):
    """
    generate linspace
    :param timesteps: the number of steps of noise addition
    :param start: starting value of beta
    :param stop: ending value of beta
    """
    return torch.linspace(start,stop,timesteps)


