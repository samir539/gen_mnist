import sys
import os.path
import urllib.request
import tarfile
import MNIST_funcs

##import sequence MNIST
host_path = "https://raw.githubusercontent.com/edwin-de-jong/mnist-digits-stroke-sequence-data/master/sequences.tar.gz"
seq_MNIST_path = "seq_MNIST" #local path to store the data
#hello
def get_seq_MNIST(seq_MNIST_host_path = host_path, loc_path = seq_MNIST_path):
    os.makedirs(seq_MNIST_path,exist_ok=True)
    tgz_file = "seq_MNIST.tgz"
    urllib.request.urlretrieve(host_path,tgz_file)
    seq_MNIST_tgz = tarfile.open(tgz_file)
    seq_MNIST_tgz.extractall(path=seq_MNIST_path)
    seq_MNIST_tgz.close() 

get_seq_MNIST()



