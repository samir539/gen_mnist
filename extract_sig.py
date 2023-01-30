from joblib import Parallel, delayed
import joblib
from MNIST_funcs import stream_normalise_mean_and_range as snmr
from MNIST_funcs import stream_normalise_mean_and_std as snms
import MNIST_funcs
import esig
import os, os.path
import errno
from pathlib import Path

#Find no. of cpu cores
N_CPU = joblib.cpu_count()

def extract_sequences():
    """
    function to extract the train/test points, train/test_inputs, train/test_targets
    #note just put these together since its a generative model
    """
    N_CPU = joblib.cpu_count()
    train_points, train_inputs, train_targets = MNIST_funcs.mnist_train_data("seq_MNIST", N_CPU)
    test_points, test_inputs, test_targets = MNIST_funcs.mnist_test_data("seq_MNIST", N_CPU)
    seq_dict = {"train_points":train_points, "train_inputs":train_inputs,"train_targets":train_targets,"test_points":test_points,"test_inputs":test_inputs,"test_targets":test_targets}
    return seq_dict


#Compute signatures
SIGNATURE_LEVEL = 8

def sigcomp(train_points,test_points,SIGNATURE_LEVEL):
    """
    compute sigs #improve this docstring
    """
    train_sigs_snmr = Parallel(n_jobs=N_CPU)([delayed(esig.stream2sig)(snmr(train_points[k]), SIGNATURE_LEVEL)
                                            for k in range(len(train_points)) ])    
    test_sigs_snmr = Parallel(n_jobs=N_CPU)([delayed(esig.stream2sig)(snmr(test_points[k]), SIGNATURE_LEVEL)
                                            for k in range(len(test_points)) ])
        
    train_sigs_snms = Parallel(n_jobs=N_CPU)([ delayed(esig.stream2sig)(snms(train_points[k]), SIGNATURE_LEVEL)
                                            for k in range(len(train_points)) ])    
    test_sigs_snms = Parallel(n_jobs=N_CPU)([ delayed(esig.stream2sig)(snms(test_points[k]), SIGNATURE_LEVEL)
                                            for k in range(len(test_points)) ])
    sig_dict = {"train_sigs_snmr": train_sigs_snmr, "train_sigs_snms":train_sigs_snms, "test_sigs_snmr":test_sigs_snmr, "test_sigs_snms":test_sigs_snms}
    return sig_dict


def write_to_text(dict):
    """
    function to put extracted lists into textfile for future use
    :param dict: dict containing the lists to put in textfile
    """
    for key,val in dict.items():
        content = '\n'.join(str(i) for i in val)
        p = Path('./Data_extracted/')
        p.mkdir(exist_ok=True)
        filename = '{}.txt'.format(key)
        if os.path.isfile(p/filename):
            print("--data already in files --") #note that if we want to change the data we need to get rid of this if/else or delete the directory/file
            return None
        else:
            with (p/filename).open('w') as opened_file:
                opened_file.writelines(content)


#store in textfiles
def store_in_txt():
    """
    store the extracted sequences in their respective textfiles in the Data_extracted repository 
    """
    data_extracted = extract_sequences()
    write_to_text(data_extracted)
    sig_extracted = extract_sequences()
    write_to_text(sig_extracted)


    
    


