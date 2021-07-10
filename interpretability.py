from dataset import load_data, load_yelp
from model import Autoencoder, Autoencoder_NonHid
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, LassoCV
from sklearn.preprocessing import OneHotEncoder
from xclib.evaluation import xc_metrics
from pydrf.order import CategoryOrderEncoder
from pydrf.model import DRFModel
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import pickle
import utils
import utils_interpret
import tensorflow as tf
import os

data_name = 'yelp'
X_train, y_train, _, _, _ = load_data(data_name, 'train')
X_test, y_test, _, _, _ = load_data(data_name, 'test')
_, _, feature_names, label_names = load_yelp('test', return_names=True)
# y_train = y_train.toarray()
# y_test = y_test.toarray()


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True, force_gpu_compatible=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.keras.backend.set_session(sess)

# Define autoencoder
input_dim = y_train.shape[1]
hid_dim = None
latent_dim = y_train.shape[1]
# Load autoencoder
autoencoder = Autoencoder_NonHid(input_dim=input_dim, latent_dim=latent_dim)
autoencoder.load_weights('AE/{}_AE_{}-{}-{}'.format(data_name, input_dim, hid_dim, latent_dim))

# Get encoded labels
latent_y_train = autoencoder.encoder.predict(y_train)
latent_y_test = autoencoder.encoder.predict(y_test)


######################
# Load forest models
trees_per_label = 3
max_features = 0.8
max_depth = 5
model_name = 'models/{}_t{}_f{}_d{}.pickle'.format(data_name, trees_per_label, max_features, max_depth)
with open(model_name, 'rb') as file:
    models = pickle.load(file)

# 將資料轉換為規則
enc_X_train = utils.transform_data(X_train, models)
enc_X_test = utils.transform_data(X_test, models)

# 對規則進行 one-hot encoding
rule_onehot_encoder = OneHotEncoder()
rule_onehot_encoder.fit(enc_X_train)
rule_table_X_train = rule_onehot_encoder.transform(enc_X_train)
rule_table_X_test = rule_onehot_encoder.transform(enc_X_test)

####### 訓練線性模型 ########
print("Training")
# lm_models = utils.train_model_per_label(rule_table_X_train, latent_y_train, LassoCV, param={'n_alphas': 30, 'cv': 3}, n_jobs=8)
lm_models = utils.train_model_per_label(rule_table_X_train, latent_y_train, Lasso,
                                        param={'alpha': 0.01}, n_jobs=8)
# lm_models = utils.train_model_per_label(rule_table_X_train, latent_y_train, LinearRegression, n_jobs=8)
# cv = LassoCV(n_alphas=20, cv=3, random_state=0).fit(rule_table_X_train, latent_y_train[:, 0])
print("Predicting")
latent_y_pred = utils.predict(rule_table_X_test, lm_models, n_jobs=16)


# 預測結果
y_pred = autoencoder.decoder.predict(latent_y_pred)
xc_metrics.ndcg(sp.sparse.csr_matrix(y_pred), y_test)
xc_metrics.precision(sp.sparse.csr_matrix(y_pred), y_test)

# 計算加權後的模型係數 (規則, 標籤)
weighted_coef = utils_interpret.return_rule_coef(lm_models, autoencoder)

# 得到每棵樹的規則數量
num_rule = utils_interpret.get_num_rule(rule_onehot_encoder)
rule_bound = utils_interpret.get_rule_bound(num_rule)


# 讀取規則
# from skmultilearn.dataset import load_dataset
# _, _, feature_names, label_names = load_dataset('bibtex', 'train', './multilabel_datasets/Bibtex/')

# 得到特徵名稱
# feature_list = [name for name, value in feature_names] # 將特徵名稱轉為特定形式
feature_list = list(feature_names)

# 計算特定標籤的係數
label_idx = 0 ### 指定標籤
# topn_coefs, topn_rules, topn_rule_idx = utils_interpret.get_rules_by_label(label_idx, rule_bound, weighted_coef, 5, models, feature_names, False, True) # 對特定標籤來說最重要的規則
# fc = utils_interpret.compute_feature_coef(label_idx, rule_bound, weighted_coef, models, feature_names)

###### 利用 bootlasso 畫出重要規則 #####
from tqdm import tqdm
from sklearn.utils import resample
# 訓練 lasso 得到各規則對標籤的係數
rw_coef = list()
for i in tqdm(range(50)):
    # Boostraping
    sub_X_train, sub_y_train = resample(rule_table_X_train, latent_y_train, 
                                        n_samples=X_train.shape[0], 
                                        random_state=i)
    # 訓練模型
    boot_lms = utils.train_model_per_label(sub_X_train, sub_y_train, Lasso, 
                                            param={'alpha': 0.01}, n_jobs=8)
    # 計算加權的係數
    coef = utils_interpret.return_rule_coef(boot_lms, autoencoder)
    rw_coef.append(coef.reshape(1, coef.shape[0], coef.shape[1]))
rw_arr = np.concatenate(rw_coef, axis=0) # 串連所有加權的係數
# rw_arr[:, :, label_idx] #指定標籤

# 畫圖
n_rule = rule_bound[-1] + 1
for label_idx in range(y_train.shape[1]):
#     coef_sort_idx = rw_arr[:, :, label_idx].mean(axis=0).argsort()[::-1][:5] #抓對特定標籤平均前 5 重要的規則 idx
    coef_sort_idx = weighted_coef[:, label_idx].argsort()[::-1]#[:topn] # 原 lasso 的係數排序
    coef_sel = weighted_coef[coef_sort_idx, label_idx] # 原 lasso 的係數
    print(label_names[label_idx])
#     for idx in coef_sort_idx: # 印出規則內容和符合的樣本數量
#         rule = utils_interpret.get_rule_by_index(idx, rule_bound, models, feature_names)
#         print('ruel_idx', idx, ':', rule)
#         t_idx = utils_interpret.get_tree_by_index(idx, rule_bound)
#         t_ord, rt_idx = utils_interpret.get_rule_order_in_tree_by_index(idx, rule_bound, models)# 第 t 棵樹的第 n 條規則
#         leaf = np.unique(enc_X_train[:, t_idx])[rt_idx]
#         flter = (enc_X_train[:, t_idx]==leaf)
#         print('# samples: ', sum(flter))
#         print("positive/all: {}/{}".format(sum(y_train.toarray()[flter, label_idx]==1), sum(flter)))
#         print()
    fig_name = "{}.svg".format(label_names[label_idx])
    # 畫圖 #utils_interpret.
    utils_interpret.plot_confidence_intervals(np.arange(n_rule), rw_arr[:, :, label_idx],
                                              topn=5,
                                              sorted_idx=coef_sort_idx, sorted_coef=coef_sel,
                                              prefix="rule ",
                                              models=models,
                                              rule_bound=rule_bound,
                                              figsize=[6, 5],
                                              save_fig=fig_name,
                                              flt="abs_both", layout='odds')

# print out rule content
idx = 141
rule = utils_interpret.get_rule_by_index(idx, rule_bound, models, feature_names)
t_idx = utils_interpret.get_tree_by_index(idx, rule_bound)
t_ord, rt_idx = utils_interpret.get_rule_order_in_tree_by_index(idx, rule_bound, models)# 第 t 棵樹的第 n 條規則
leaf = np.unique(enc_X_train[:, t_idx])[rt_idx]
flter = (enc_X_train[:, t_idx]==leaf)
print("T{}.R{}".format(t_idx, rt_idx))
print('# samples: ', sum(flter))
print('ruel_idx', idx, ':', rule)


    
    
    
    
    
    
    
# # # # # 列出重要的規則與規則內包含的樣本數量
# 排出前幾個重要的特徵與他的總係數
# print("="*5, "label:", label_names[label_idx], "="*5)
# for idx in fc.argsort()[::-1][:10]:
#     print("{:.4f} {}".format(fc[idx], feature_list[idx]))


# rule_idx = topn_rule_idx[0]
# rule = utils_interpret.get_rule_by_index(rule_idx, rule_bound, models, None)
# t_idx = utils_interpret.get_tree_by_index(rule_idx, rule_bound)
# # m_idx = utils_interpret.get_model_by_index(rule_idx, rule_bound, models)
# t_ord, rt_idx = utils_interpret.get_rule_order_in_tree_by_index(rule_idx, rule_bound, models)# 第 t 棵樹的第 n 條規則

# leaf = np.unique(enc_X_train[:, t_idx])[rt_idx]
# flter = (enc_X_train[:, t_idx]==leaf) # 屬於指定葉節點的資料點


# 列出對特定標籤來說重要的規則，以及符合該規則的樣本數量
# for label_idx in range(5):
#     print("label :", label_names[label_idx])
#     topn_coefs, topn_rules, topn_rule_idx = utils_interpret.get_rules_by_label(label_idx, rule_bound, weighted_coef, 5, models, feature_names, False, True) # 對特定標籤來說最重要的規則

#     for coef, rule, rule_idx in zip(topn_coefs, topn_rules, topn_rule_idx):
#         t_idx = utils_interpret.get_tree_by_index(rule_idx, rule_bound)
#         t_ord, rt_idx = utils_interpret.get_rule_order_in_tree_by_index(rule_idx, rule_bound, models)
#         leaf = np.unique(enc_X_train[:, t_idx])[rt_idx]
#         flter = (enc_X_train[:, t_idx]==leaf)
#         print("rule_id:", rule_idx)
#         print("coefficients: {:.4f}".format(coef))
#         print(rule)
#         print("# samples:", sum(flter), "\n")
    
    


# 用 bootlasso 估計所有標籤的特徵係數，並畫出信賴區間
topn = 5
for label_idx in range(4,5):
    print("label :", label_names[label_idx])
    topn_coefs, topn_rules, topn_rule_idx = utils_interpret.get_rules_by_label(label_idx, 
                                                                               rule_bound,
                                                                               weighted_coef,
                                                                               3, models,
                                                                               feature_names, 
                                                                               False)
    
    # 對規則分別找出相應的群組，並畫出信賴區間
    for coef, rule, rule_idx in zip(topn_coefs, topn_rules, topn_rule_idx):
        t_idx = utils_interpret.get_tree_by_index(rule_idx, rule_bound)
        t_ord, rt_idx = utils_interpret.get_rule_order_in_tree_by_index(rule_idx, rule_bound, models)
        leaf = np.unique(enc_X_train[:, t_idx])[rt_idx] # 得出相應的規則
        flter = (enc_X_train[:, t_idx]==leaf)
        print("T{}.R{}".format(t_idx, rt_idx))
        print(sum(flter))
        ls_flt = utils.train_model_per_label(rule_table_X_train[flter, :],
                                             latent_y_train[flter, :], 
                                             Lasso,
                                             param={'alpha': 0.01}, n_jobs=8)
        flt_coef = utils_interpret.return_rule_coef(ls_flt, autoencoder)
        flt_coef_sort_idx = flt_coef[:, label_idx].argsort()[::-1] #[:topn]
        flt_coef_sort_sel = flt_coef[flt_coef_sort_idx, label_idx]
        boot_coef = utils_interpret.compute_bootstrap_lasso_coef(X_train, latent_y_train, 
                                                                 flter, label_idx, 
                                                                 autoencoder=autoencoder,
                                                                 sample_times=5)
    
#         flt_coef = boot_coef.mean(axis=0) # 抓對特定標籤平均前 5 重要的規則 index
#         flt_coef_sort_idx = flt_coef.argsort()[::-1] #[:topn]
#         flt_coef_sort_sel = flt_coef[flt_coef_sort_idx]
#         utils_interpret.plot_confidence_intervals(feature_names, boot_coef)
        utils_interpret.plot_confidence_intervals(feature_names, boot_coef,
                                                  sorted_idx=flt_coef_sort_idx,
                                                  sorted_coef=flt_coef_sort_sel, 
                                                  flt='negative'
                                 )



# label_idx = 4
# # 挑選出重要的規則
# coef_sort_idx = rw_arr[:, :, label_idx].mean(axis=0).argsort()[::-1][:5]
# print(label_names[label_idx])
# # 對每條規則畫出重要變數的信賴區間
# for rule_idx in coef_sort_idx:
#     # 得出特定規則與符合該規則的樣本
#     t_idx = utils_interpret.get_tree_by_index(rule_idx, rule_bound)
#     t_ord, rt_idx = utils_interpret.get_rule_order_in_tree_by_index(rule_idx, rule_bound, models)# 第 t 棵樹的第 n 條規則
#     rule = utils_interpret.get_rule_by_index(rule_idx, rule_bound, models, feature_names)
#     leaf = np.unique(enc_X_train[:, t_idx])[rt_idx]
#     flter = (enc_X_train[:, t_idx]==leaf)
    
#     fw_coef = list()
#     # 利用規則訓練 lasso 模型計算變數重要性
#     for i in tqdm(range(10)):
#         sub_X_train, sub_y_train = resample(X_train[flter], latent_y_train[flter],
#                                             n_samples=X_train.shape[0], 
#                                             random_state=i)
        
#         # 訓練模型
#         lm_models = utils.train_model_per_label(sub_X_train, sub_y_train, Lasso, 
#                                                 param={'alpha': 0.01}, n_jobs=8)
        
#         # 計算加權的係數
#         coef = utils_interpret.return_rule_coef(lm_models, autoencoder)
#         fw_coef.append(coef.reshape(1, coef.shape[0], coef.shape[1]))
#     # 計算
#     fw_arr = np.concatenate(fw_coef, axis=0)
#     n_rule = rule_bound[-1] + 1
# #     coef_sort_idx = fw_arr[:, :, label_idx].mean(axis=0).argsort()[::-1][:5] 
    
#     #印出規則與畫圖
#     print(rule)
#     utils_interpret.plot_confidence_intervals(feature_names, fw_arr[:, :, label_idx], 5)


###########
# # 對特定標籤計算bootlasso
# lasso_coefs = []

# topn=5
# test_flter = (enc_X_test[:, t_idx]==leaf)
# sub_X_test = X_test.toarray()[test_flter]
# sub_y_test = y_test[test_flter, label_idx].reshape(1, -1)

# for i in tqdm(range(3)): #抽樣次數
#     sub_X_train, sub_y_train = resample(X_train[flter], y_train[flter, label_idx], n_samples=X_train.shape[0], random_state=i) # 取出 subgroup 並重複抽樣到原始的樣本數量

#     lasso = Lasso(alpha=0.01) # 訓練模型
#     lasso.fit(sub_X_train.toarray(), sub_y_train.toarray())
#     pre = lasso.predict(sub_X_test)
#     print(xc_metrics.ndcg(sp.sparse.csr_matrix(pre), sub_y_test))
#     lasso_coefs.append(lasso.coef_.reshape(1, -1)) # 得到係數

# lc = np.concatenate(lasso_coefs, axis=0) # 串接係數
# coef_sort_idx = lc.mean(axis=0).argsort()[::-1][:topn] #抓對特定標籤平均前 5 重要的係數
# fn = list(map(lambda idx: feature_names[idx], coef_sort_idx)) # 重要的特徵名稱
# topn_bootstrap_coef = lc[:, coef_sort_idx]


# # Confidence intervals
# intervals = [np.percentile(topn_bootstrap_coef,  2.5, axis=0), 
#              np.percentile(topn_bootstrap_coef, 97.5, axis=0)] # 計算 95% 信賴區間




# # # Visualize
# plt.ylabel('feature')
# plt.xlabel('coefficient')
# # viridis = plt.cm.get_cmap('viridis', len(intervals[0]))
# plt.vlines(0, ymin=0, ymax=len(intervals[0])-0.5, linestyles='dashed')
# # plt.hlines(0, xmin=0, xmax=len(intervals[0])-0.5, linestyles='dashed')
# plt.plot(topn_bootstrap_coef.mean(axis=0), range(topn), '|k')
# # plt.plot(coef_sel, '_k')
# # prefix = "rule"+" "
# prefix = ""
# ticks = [prefix + name for name in fn]
# plt.yticks(range(5), ticks)
# for idx, (min_int, max_int) in enumerate(zip(intervals[0], intervals[1])):
#     plt.hlines(y=idx, xmin=min_int, xmax=max_int, label=fn[idx])
# # plt.legend()



####
# rule, depth = utils.extract_rule(models[0].estimators_[0], 1, feature_names=feature_list)
# print(rule)

# # 萃取規則與得到全部節點深度
# rule, depth = utils.extract_rule(models[0].estimators_[0], 3, feature_names=feature_list)
# rule, depth = utils.extract_rule(models[4].estimators_[1], 6, feature_names=feature_list)

# import seaborn as sns
# sns.displot(fc, kind="kde")
