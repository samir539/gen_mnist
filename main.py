
import numpy as np
from joblib import Parallel, delayed
from MNIST_funcs import stream_normalise_mean_and_range as snmr
from MNIST_funcs import stream_normalise_mean_and_std as snms
from get_data import get_seq_MNIST
import esig

##load in data
get_seq_MNIST()