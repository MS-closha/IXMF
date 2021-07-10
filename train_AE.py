from skmultilearn.dataset import load_dataset
from model import Autoencoder, Autoencoder_NonHid
from tensorflow.keras import regularizers
from xclib.evaluation import xc_metrics
from xclib.data import data_utils
from dataset import load_data
from utils import reconstruction_error, sparsity
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy as sp
import numpy as np
import argparse
import os

def parse_arg():
    parser = argparse.ArgumentParser(description="Train Autoencoders")
    parser.add_argument('--dataname', '-d', type=str, help='name of dataset used', default='bibtex')
    parser.add_argument('--epochs', '-e', type=int, help='#epochs', default=3)
    parser.add_argument('--lr', '-l', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--latent', '-c', type=int, default=None, help='dimension of latent code')
    return parser.parse_args()

if __name__ == '__main__':
    arg = parse_arg()
    
    # ~ data name ~
    data_name = arg.dataname
    # Load data
    X_train, y_train, _, _, _ = load_data(data_name, 'train')
    X_test, y_test, _, _, _ = load_data(data_name, 'test')

    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True, force_gpu_compatible=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(sess)


    # Define autoencoder

    if True:
#     if data_name == 'eurlex' or data_name == 'delicious': # the eurlex needs less complicated structure
        input_dim = y_train.shape[1]
        # ~ dimension ~
        hid_dim = None
        latent_dim = y_train.shape[1] if (arg.latent is None) else arg.latent
#         latent_dim = y_train.shape[1]
        autoencoder = Autoencoder_NonHid(input_dim, latent_dim)
    else:
        input_dim = y_train.shape[1]
        # ~ dimension ~
        hid_dim = y_train.shape[1]
        latent_dim = y_train.shape[1]
        autoencoder = Autoencoder(input_dim, hid_dim, latent_dim)

    autoencoder.compile(optimizer='Adam',
                        loss="categorical_crossentropy",
                        metrics=['accuracy'], 
                        lr=arg.lr)

    # ~ training epochs ~
    e = arg.epochs

    # Train autoencoder
#     callbacks = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5) # earlystopping
    history = autoencoder.fit(
                    y_train.toarray().astype('float32'), 
                    y_train.toarray().astype('float32'),
                    validation_data=[y_test, y_test],
                    epochs=e,
                    batch_size=32,
                    shuffle=True,
#                     callbacks=[callbacks]
    )

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1, e+1)

    # Create a directory
    os.makedirs('ae_figure', exist_ok=True)
    
    # Plot the outcome
    plt.figure()
    plt.title('Loss')
    plt.plot(epochs, train_loss)
    plt.plot(epochs, val_loss)
    plt.legend(['train_loss', 'test_loss'])
    plt.savefig('ae_figure/{}_loss'.format(data_name))

    plt.figure()
    plt.title('Accuracy')
    plt.plot(epochs, train_acc)
    plt.plot(epochs, val_acc)
    plt.legend(['train_acc', 'test_acc'])
    plt.savefig('ae_figure/{}_acc'.format(data_name))


    # Encode and decode data with auto-encoder
    latent_y_train = autoencoder.encoder.predict(y_train)
    latent_y_test = autoencoder.encoder.predict(y_test)
    rec_train = autoencoder.decoder.predict(latent_y_train)
    rec_test = autoencoder.decoder.predict(latent_y_test)

    # Compute reconstruction error
    print("Reconstruction error for training: ", reconstruction_error(rec_train, y_train))
    print("Reconstruction error for testing: ", reconstruction_error(rec_test, y_test))

    rec_train = sp.sparse.csr_matrix(rec_train)
    rec_test = sp.sparse.csr_matrix(rec_test)
    print("Reconstruction nDCG for training: ", xc_metrics.ndcg(rec_train, y_train))
    print("Reconstruction nDCG for testing: ", xc_metrics.ndcg(rec_test, y_test))


    # Save weights
    autoencoder.save_weights('AE/{}_AE_{}-{}-{}'.format(data_name, input_dim, hid_dim, latent_dim))


