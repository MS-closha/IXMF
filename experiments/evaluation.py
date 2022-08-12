from model import Autoencoder_NonHid, Autoencoder
from xclib.evaluation import xc_metrics
from xclib.data import data_utils
from dataset import load_data
from sklearn.model_selection import train_test_split
import tensorflow as tf
import seaborn as sns
import scipy as sp
import numpy as np
import pickle
import utils
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True, force_gpu_compatible=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)

# path = "./multilabel_datasets/Bibtex"
# X, y, num_samples, num_features, num_labels = data_utils.read_data('{}/{}_data.txt'.format(path, data_name.capitalize()), header=True, dtype='float32', zero_based=True)
# indx = np.arange(X.shape[0])
# train_indx = np.random.choice(indx, 4880, replace=False)
# test_indx = np.setdiff1d(indx, train_indx)
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=4880)

# ~ data name ~
data_name = 'yelp'

# Load data
X_train, y_train, _, _, _ = load_data(data_name, 'train')
X_test, y_test, _, _, _ = load_data(data_name, 'test')

utils.sparsity(y_train.toarray())
# if data_name == 'eurlex' or data_name == 'delicious': # the eurlex needs less complicated structure
#     input_dim = y_train.shape[1]
#     hid_dim = None
#     latent_dim = y_train.shape[1]
#     autoencoder = Autoencoder_NonHid(input_dim, latent_dim)
# else: 
#     input_dim = y_train.shape[1]
#     hid_dim = y_train.shape[1]
#     latent_dim = y_train.shape[1]
#     autoencoder = Autoencoder(input_dim, hid_dim, latent_dim)

input_dim = y_train.shape[1]
hid_dim = None
latent_dim = y_train.shape[1]
autoencoder = Autoencoder_NonHid(input_dim, latent_dim)    

# Load weights of AE
autoencoder.load_weights('../process/AEs/{}_AE_{}-{}-{}'.format(data_name, input_dim, hid_dim, latent_dim))


# ~ model parameter for loading ~
trees_per_label = 3
max_features = 0.8
max_depth = 5

model_name = '../process/models/{}_t{}_f{}_d{}.pickle'.format(data_name, trees_per_label, max_features, max_depth)
# Load forest models
with open(model_name, 'rb') as file:
    models = pickle.load(file)

# Predict
n_jobs = 16
print("Predict outcome...")
latent_y_train_pred = utils.predict(X_train, models, n_jobs)
latent_y_test_pred = utils.predict(X_test, models, n_jobs) # use X_test to predict the outcome
y_train_pred = autoencoder.decoder.predict(latent_y_train_pred)
y_test_pred = autoencoder.decoder.predict(latent_y_test_pred) # decode the output value with auto-encoder


# Evaluate
y_train_pred = sp.sparse.csr_matrix(y_train_pred) # transform to csr_matrix for evaluation
y_test_pred = sp.sparse.csr_matrix(y_test_pred)

p5_train = xc_metrics.precision(y_train_pred, y_train) # call function
n5_train = xc_metrics.ndcg(y_train_pred, y_train)
p5 = xc_metrics.precision(y_test_pred, y_test)
n5 = xc_metrics.ndcg(y_test_pred, y_test)

# Print out
print("===== Train =====")
print("Precision@k:", p5_train)
print("nDCG@k:", n5_train)

print("\n===== Test =====")
print("Precision@k:", p5)
print("nDCG@k:", n5)


# FastXML


    
# # explanation
# imp = np.concatenate(list(map(lambda m: m.feature_importances_.reshape(-1,1), models)), axis=1) # 計算每個 latent 的特徵重要性
# weighted_imp = utils.return_feature_imp(models, autoencoder)# 計算加權後的特徵重要性


# # 每個 label 找出前 10 個重要的特徵
# for i, label in enumerate(label_names):
#     print('======== {} {} ========'.format(i, label[0]))
#     for j in weighted_imp.argsort(axis=0)[-10:][::-1].T[i]:
#         print('{} {:>16s}: {:.6f}'.format(j, feature_names[j][0], weighted_imp[j][i]))

# # 每個 latent 找出前 10 個重要的特徵
# for n, m in enumerate(models):
#     imp = models[n].feature_importances_
#     print('='*10 + " model {} ".format(n), '='*10)
#     for i in imp.argsort()[-10:][::-1]:
#         print('{}: {:.4f}%'.format(feature_names[i][0], imp[i]*100))


# # 印出各特徵的密度圖
# for i, _ in enumerate(imp):
#     sns.displot(weighted_imp[i], kde=True)

