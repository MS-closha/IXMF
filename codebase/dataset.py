from skmultilearn.dataset import load_dataset
from xclib.data import data_utils
from scipy.sparse import csr_matrix
from scipy.io import arff
import pandas as pd
import numpy as np
import os

def load_data(data_name, variant, path='../multilabel_datasets'):
    path = os.path.join(path, data_name.capitalize())
    
    if data_name in ['bibtex', 'delicious', 'mediamill']:
        X, y, num_samples, num_features, num_labels = read_data(data_name, variant, path)
    
    elif data_name == 'eurlex':
        X, y, num_samples, num_features, num_labels = data_utils.read_data('{}/eurlex_{}.txt'.format(path, variant))
    
    elif data_name == 'yelp':
        X, y, num_samples, num_features, num_labels = load_yelp(variant, path)
        
    else:
        raise ValueError("No dataset named '{}'".format(data_name))
        
    return (X, y, num_samples, num_features, num_labels)


def read_data(data_name, variant, path):
    if variant == 'train':
        indx = data_utils.read_split_file('{}/{}_trSplit.txt'.format(path, data_name))

    elif variant == 'test':
        indx = data_utils.read_split_file('{}/{}_tstSplit.txt'.format(path, data_name))
        
    else:
        raise ValueError("variant should be either 'train' or 'test', but got '{}'".format(variant))

    X, y, num_samples, num_features, num_labels = data_utils.read_data('{}/{}_data.txt'.format(path, data_name.capitalize()), header=True, dtype='float32', zero_based=True)
    
    X_split = data_utils._split_data(X, indx[:,0]-1)
    y_split = data_utils._split_data(y, indx[:,0]-1)

    return (X_split, y_split, X_split.shape[0], X_split.shape[1], y_split.shape[1])


def load_yelp(variant, path='./multilabel_datasets/Yelp', return_names=False):
    if variant not in ('train', 'test'):
        raise ValueError("variant should be either 'train' or 'test', but got '{}'".format(variant))
        
    data, meta = arff.loadarff('{}/yelp{}.arff'.format(path, variant.capitalize()))
    
    df = pd.DataFrame(data, dtype=np.float32)
    X = csr_matrix(df.iloc[:, 0:-8])
    y = csr_matrix(df.iloc[:, -8:-3])

    if return_names:        
        feature_names = df.iloc[:, 0:-8].columns
        label_names = df.iloc[:, -8:-3].columns
        return (X, y, feature_names, label_names)
    else:
        return(X, y, X.shape[0], X.shape[1], y.shape[1])
