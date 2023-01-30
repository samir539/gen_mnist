import torch
import torch.nn.functional as F
import matplotlib as plt

#we need to 
# read in signatures
# apply noise

def get_beta(timesteps,start=0.0001,stop=0.02):
    """
    generate linspace
    :param timesteps: the number of steps of noise addition
    :param start: starting value of beta
    :param stop: ending value of beta
    """
    return torch.linspace(start,stop,timesteps)

#based ib x0 (the inital value of the signature) we wish to get the noisy verson for any arbirary time step t
#q(x_t | x_0) = N(x_t; sqrt(alpha_hat_t)*x_0,(1-alpha_hat_t)I)

#beta value schedule
T = 200
beta_vals = get_beta(200)



