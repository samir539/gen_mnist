import sys
import os.path
import urllib.request
from joblib import Parallel, delayed
import tarfile
import MNIST_funcs
import joblib
from MNIST_funcs import stream_normalise_mean_and_range as snmr
from MNIST_funcs import stream_normalise_mean_and_std as snms
import esig

##import sequence MNIST
host_path = "https://raw.githubusercontent.com/edwin-de-jong/mnist-digits-stroke-sequence-data/master/sequences.tar.gz"
seq_MNIST_path = "seq_MNIST" #local path to store the data

def get_seq_MNIST(seq_MNIST_host_path = host_path, loc_path = seq_MNIST_path):
    os.makedirs(seq_MNIST_path,exist_ok=True)
    tgz_file = "seq_MNIST.tgz"
    urllib.request.urlretrieve(host_path,tgz_file)
    seq_MNIST_tgz = tarfile.open(tgz_file)
    seq_MNIST_tgz.extractall(path=seq_MNIST_path)
    seq_MNIST_tgz.close() 

# get_seq_MNIST()

#Extract sequences
N_CPU = joblib.cpu_count()
train_points, train_inputs, train_targets = MNIST_funcs.mnist_train_data(seq_MNIST_path, N_CPU)
test_points, test_inputs, test_targets = MNIST_funcs.mnist_test_data(seq_MNIST_path, N_CPU)


#Compute signatures 
SIGNATURE_LEVEL = 8

train_sigs_snmr = Parallel(n_jobs=N_CPU)([delayed(esig.stream2sig)(snmr(train_points[k]), SIGNATURE_LEVEL)
                                          for k in range(len(train_points)) ])    
test_sigs_snmr = Parallel(n_jobs=N_CPU)([delayed(esig.stream2sig)(snmr(test_points[k]), SIGNATURE_LEVEL)
                                         for k in range(len(test_points)) ])
    
train_sigs_snms = Parallel(n_jobs=N_CPU)([ delayed(esig.stream2sig)(snms(train_points[k]), SIGNATURE_LEVEL)
                                          for k in range(len(train_points)) ])    
test_sigs_snms = Parallel(n_jobs=N_CPU)([ delayed(esig.stream2sig)(snms(test_points[k]), SIGNATURE_LEVEL)
                                         for k in range(len(test_points)) ])

print(test_sigs_snmr)





