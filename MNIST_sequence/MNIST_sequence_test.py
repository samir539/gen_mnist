import sys
import os.path
import urllib.request
import tarfile
import MNIST_funcs

##import sequence MNIST
host_path = "https://edwin-de-jong.github.io/blog/mnist-sequence-data/sequences.tar.gz"
seq_MNIST_path = "seq_MNIST" #local path to store the data

def get_seq_MNIST(seq_MNIST_host_path = host_path, loc_path = seq_MNIST_path):
    os.makedirs(seq_MNIST_path,exist_ok=True)
    tgz_path = "seq_MNIST.tgz"
    urllib.request.urlretrieve(host_path,tgz_path)
    seq_MNIST_tgz = tarfile.open(tgz_path)
    seq_MNIST_tgz.extractall(path=seq_MNIST_path)
    seq_MNIST_tgz.close() 




