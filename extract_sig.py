from joblib import Parallel, delayed
import joblib
from MNIST_funcs import stream_normalise_mean_and_range as snmr
from MNIST_funcs import stream_normalise_mean_and_std as snms
import MNIST_funcs
import esig
import os, os.path
import errno
from pathlib import Path

#Extract sequences
N_CPU = joblib.cpu_count()
def extract_sequences():
    """
    function to extract the train/test points, train/test_inputs, train/test_targets
    #note just put these together since its a generative model
    """
    N_CPU = joblib.cpu_count()
    train_points, train_inputs, train_targets = MNIST_funcs.mnist_train_data("seq_MNIST", N_CPU)
    test_points, test_inputs, test_targets = MNIST_funcs.mnist_test_data("seq_MNIST", N_CPU)
    return train_points,train_inputs,train_targets,test_points,test_inputs,test_targets

#Write into textfiles
def write_to_text(input_list):
    """
    function to put extracted list into textfile for future use
    :param input_list: list to put in textfile
    """
    content = '\n'.join(str(i) for i in input_list)
    p = Path('./Data/')
    p.mkdir(exist_ok=True)
    filename = '{}.txt'.format(input_list)
    with (p/filename).open('w') as opened_file:
        opened_file.writelines(content)


#Compute signatures
SIGNATURE_LEVEL = 8

def sigcomp(train_points,test_points,SIGNATURE_LEVEL):
    train_sigs_snmr = Parallel(n_jobs=N_CPU)([delayed(esig.stream2sig)(snmr(train_points[k]), SIGNATURE_LEVEL)
                                            for k in range(len(train_points)) ])    
    test_sigs_snmr = Parallel(n_jobs=N_CPU)([delayed(esig.stream2sig)(snmr(test_points[k]), SIGNATURE_LEVEL)
                                            for k in range(len(test_points)) ])
        
    train_sigs_snms = Parallel(n_jobs=N_CPU)([ delayed(esig.stream2sig)(snms(train_points[k]), SIGNATURE_LEVEL)
                                            for k in range(len(train_points)) ])    
    test_sigs_snms = Parallel(n_jobs=N_CPU)([ delayed(esig.stream2sig)(snms(test_points[k]), SIGNATURE_LEVEL)
                                            for k in range(len(test_points)) ])
    return train_sigs_snmr,train_sigs_snms,test_sigs_snmr,test_sigs_snms

