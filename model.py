from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow as tf


class Autoencoder(Model):
    def __init__(self, input_dim, hid_dim, latent_dim):
        super(Autoencoder, self).__init__()  
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_dim),
            layers.Dense(hid_dim, activation='linear'),
            layers.Dense(latent_dim, activation='linear'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim, )),
            layers.Dense(hid_dim, activation='linear'),
            layers.Dense(input_dim, activation='sigmoid'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class Autoencoder_NonHid(Model):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder_NonHid, self).__init__()  
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_dim),
            layers.Dense(latent_dim, activation='linear'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim, )),
            layers.Dense(input_dim, activation='sigmoid'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
