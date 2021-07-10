from xclib.evaluation import xc_metrics
from model import Autoencoder_NonHid, Autoencoder
from dataset import load_data
import argparse
import tensorflow as tf
import scipy as sp
import pickle
import utils
import time
import csv
import os

def parse_arg():
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument('--dataname', '-d', type=str, help='name of dataset used', default='bibtex')
    parser.add_argument('--trees-per-label', '-t', type=int, help='#trees', default=3)
    parser.add_argument('--max_features', '-f', type=float, help='max number of features', default=None)
    parser.add_argument('--max_depth', '-p', type=int, help='max depth of each tree', default=None)
    parser.add_argument('--n_jobs', '-j', type=int, help='#cores', default=2)
    parser.add_argument('--latent', '-c', type=int, default=None, help='dimension of latent code')
    return parser.parse_args()


if __name__ == '__main__':
    arg = parse_arg()
    
    # GPU Setting
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.1, allow_growth = True, force_gpu_compatible = True)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
    tf.keras.backend.set_session(sess)


    # ~ data name ~
    data_name = arg.dataname # 'bibtex'
    # Load data
    X_train, y_train, _, _, _ = load_data(data_name, 'train')
    X_test, y_test, _, _, _ = load_data(data_name, 'test')


    # Define autoencoder
    if True:
#     if data_name == 'eurlex' or data_name == 'delicious': # the eurlex needs less complicated structure
        input_dim = y_train.shape[1]
        hid_dim = None
        latent_dim = y_train.shape[1] if (arg.latent is None) else arg.latent
#         latent_dim = y_train.shape[1]
        autoencoder = Autoencoder_NonHid(input_dim, latent_dim)
    else: 
        input_dim = y_train.shape[1]
        hid_dim = y_train.shape[1]
        latent_dim = y_train.shape[1]
        autoencoder = Autoencoder(input_dim, hid_dim, latent_dim)

    # Load weights of AE
    autoencoder.load_weights('AE/{}_AE_{}-{}-{}'.format(data_name, input_dim, hid_dim, latent_dim))

    # Encode the labelset with auto-encoder
    latent_y_train = autoencoder.encoder.predict(y_train.toarray().astype('float32'))
    latent_y_test = autoencoder.encoder.predict(y_test.toarray().astype('float32'))

    
    # ~ parameters ~
    trees_per_label = arg.trees_per_label
    max_features = arg.max_features
    max_depth = arg.max_depth
    n_jobs = arg.n_jobs

    # Train models
    print("- Model Training...")
    start_time = time.time()
    models = utils.train_RF_per_label(X_train, latent_y_train,  # use X_train and encoded labelset latent_y_train
                                      trees_per_label=trees_per_label, 
                                      max_features=max_features, 
                                      max_depth=max_depth,
                                      n_jobs=n_jobs)
    end_time = time.time()
    cost_time = round(end_time-start_time, 2)
    print("Time cost for training: ", cost_time) # report cost time

    # Predict
    print("- Predict outcome...")
    latent_y_train_pred = utils.predict(X_train, models, n_jobs) # use X_train to predict the outcome
    y_train_pred = autoencoder.decoder.predict(latent_y_train_pred) # decode the output value with auto-encoder
    latent_y_test_pred = utils.predict(X_test, models, n_jobs) # use X_test to predict the outcome
    y_test_pred = autoencoder.decoder.predict(latent_y_test_pred) # decode the output value with auto-encoder

    # Evaluate
    y_train_pred = sp.sparse.csr_matrix(y_train_pred)
    y_test_pred = sp.sparse.csr_matrix(y_test_pred) # transform to csr_matrix for evaluation
    p = xc_metrics.precision(y_train_pred, y_train)
    n = xc_metrics.ndcg(y_train_pred, y_train)
    
    p5 = xc_metrics.precision(y_test_pred, y_test)
    n5 = xc_metrics.ndcg(y_test_pred, y_test)
    numOfLeaf = utils.count_leaf(models[0].estimators_[0]) # count the number of leaf nodes of the first tree
    print("Train:")
    print('ndcg@k : {},\nprecision@k : {}'.format(n, p))
    print("Test:")
    print('ndcg@k : {},\nprecision@k : {}'.format(n5, p5))
    print('number of rules of tree[0]: {}'.format(numOfLeaf))

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('outcome', exist_ok=True)
    
    # Save models as pickle file
    print("- Save outcome...")
    with open('models/{}_l{}_t{}_f{}_d{}.pickle'.format(data_name, latent_dim, trees_per_label, max_features, max_depth),'wb') as file:
        pickle.dump(models, file)

    # Save evaluation as csv file
    with open('outcome/{}.csv'.format(data_name), 'a') as file:
        writer = csv.DictWriter(file, fieldnames=['latent_dim',
                                                  'numOfLeaf', 
                                                  'trees_per_label', 
                                                  'max_features',
                                                  'max_depth',
                                                  'ndcg@k', 'p@k',
                                                  'training_time'])
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow({'latent_dim': latent_dim,
                         'numOfLeaf': numOfLeaf, # write information
                         'trees_per_label': trees_per_label,
                         'max_features': max_features,
                         'max_depth': max_depth,
                         'ndcg@k': n5, 'p@k': p5,
                         'training_time': cost_time})
    print("Finished.")