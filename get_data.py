import sys
import os.path
import urllib.request
import tarfile

##import sequence MNIST
host_path = "https://raw.githubusercontent.com/edwin-de-jong/mnist-digits-stroke-sequence-data/master/sequences.tar.gz"
seq_MNIST_path = "seq_MNIST" #local path to store the data

def get_seq_MNIST(seq_MNIST_host_path = host_path, loc_path = seq_MNIST_path):
    if os.path.exists(seq_MNIST_path):
        print("--- data already loaded ---") #this is not necessarily true just means the directory exists
        return None
    else:
        os.makedirs(seq_MNIST_path,exist_ok=True)
        tgz_file = "seq_MNIST.tgz"
        urllib.request.urlretrieve(host_path,tgz_file)
        seq_MNIST_tgz = tarfile.open(tgz_file)
        seq_MNIST_tgz.extractall(path=seq_MNIST_path)
        seq_MNIST_tgz.close() 