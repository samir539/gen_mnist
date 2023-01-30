from joblib import Parallel, delayed
import joblib
from MNIST_funcs import stream_normalise_mean_and_range as snmr
from MNIST_funcs import stream_normalise_mean_and_std as snms
import MNIST_funcs
import esig

#Extract sequences
N_CPU = joblib.cpu_count()
train_points, train_inputs, train_targets = MNIST_funcs.mnist_train_data("seq_MNIST", N_CPU)
test_points, test_inputs, test_targets = MNIST_funcs.mnist_test_data("seq_MNIST", N_CPU)

#Write into textfiles

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

