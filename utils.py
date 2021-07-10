from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from joblib import Parallel, delayed
import tensorflow as tf
import numpy as np
import scipy as sp
import time
import os

def reconstruction_error(X, X_re):
    return np.sqrt(np.sum(np.square(X - X_re)))

def sparsity(X):
    return 1.0 - np.count_nonzero(X)/float(X.size)

def train_model_per_label(X_train, y_train, model, param={}, n_jobs=1):
    '''
    Train models for each label to handle multilabel data.
    '''
    models = []
    y_train_T = y_train.T # y_train_T[i] == values of each label i
    for i in range(y_train.shape[1]): # build # trees per label
        models.append(model(**param)) # append the model to the list
    fitted_models = Parallel(n_jobs = n_jobs)(delayed(model.fit)(X_train, y_train_T[i])for i, model in enumerate(models))
    return fitted_models


# Function for RF models
def train_RF_per_label(X_train, y_train, trees_per_label=3, max_features=None, max_depth=None, n_jobs=8):
    '''
    Train RandomForestRegressor models for each label to handle multilabel data.
        
    Parameters
    ----------
    X_train : ndarray or csr_matrix, required
        Training data feature.
        
    y_train : ndarray, required
        Training data labelset.
        
    trees_per_label : int
        The number of tree in the RandomForestRegressor for each label.
    
    max_features : integer, float or None
        Reference to max_features of scikit-learn RandomForestRegressor.
        
    max_depth : integer, float or None
        Reference to max_features of scikit-learn RandomForestRegressor.
        
    Returns
    -------
    fitted_models : list
    '''
    models = []
    y_train_T = y_train.T # y_train_T[i] == values of each label i
    for i in range(y_train.shape[1]): # build # trees per label
        models.append(RandomForestRegressor(n_estimators=trees_per_label, 
                                            max_features=max_features, 
                                            max_depth=max_depth, 
                                            random_state=i,
                                            n_jobs=n_jobs)) # append the RandomForestRegressor to the list
    # Use features and the ith label to fit the ith model
    fitted_models = Parallel(n_jobs = n_jobs)(delayed(model.fit)(X_train, y_train_T[i]) for i, model in enumerate(models))
    return fitted_models


def train_DRF_per_label(enc_X_train, y_train, trees_per_label=3, max_features=None, max_depth=None, n_jobs=8):
    models = []
    y_train_T = y_train.T
    for i in range(y_train.shape[1]):
        models.append(RandomForestRegressor(n_estimators=trees_per_label, 
                                            max_features=max_features, 
                                            max_depth=max_depth, 
                                            n_jobs=n_jobs))
    tree_pre_layer_label = enc_X_train.shape[1] // y_train.shape[1]
    fitted_models = Parallel(n_jobs = n_jobs)(delayed(model.fit)(enc_X_train[:, tree_pre_layer_label*i : tree_pre_layer_label*i+tree_pre_layer_label], y_train_T[i]) for i, model in enumerate(models))
#     for i in range(0, y_train_T):
#         models[i].train(enc_X_train_T[3*i : 3*i+3], y_train_T)
    return fitted_models


def DRF_predict(enc_X, models, trees_per_label, n_jobs=2):

    pred_list = []
    pred_list = Parallel(n_jobs=n_jobs)(delayed(model.predict)(enc_X[:, trees_per_label*i : trees_per_label*i+trees_per_label]) for i, model in enumerate(models))
    for i, _ in enumerate(pred_list): # reshape the prediction to concatenate
        pred_list[i] = pred_list[i].reshape(enc_X.shape[0], 1)
    pred = np.concatenate(pred_list, axis=1)
    return pred


def DRF_transform_data(enc_X, models, trees_per_label, n_jobs=2):
    enc_list = []
    enc_list = Parallel(n_jobs=n_jobs)(delayed(model.apply)(enc_X[:, trees_per_label*i : trees_per_label*i+trees_per_label])for i, model in enumerate(models))
    enc_data = np.concatenate(enc_list, axis=1)
    return enc_data


# Function for leaf encoding
def transform_data(X, models, n_jobs=2):
    '''
    Leaf encoding with RF models
    '''
    enc_list = []
    enc_list = Parallel(n_jobs=n_jobs)(delayed(model.apply)(X)for model in models)# get encoded data of each label and store in a list ( n_estimators of a RF for each label )
    enc_data = np.concatenate(enc_list, axis=1) # concatenate all the encoded data in the list ( # of columns == n_estimators*n_labels )
    return enc_data


def predict(X, models, n_jobs=2):
    '''
    Predict and concatenate the outcomes with model in the list models.
        
    Parameters
    ----------
    X : ndarray, required
        Data feature.
        
    models : list of forest models, required
    
    n_jobs : integer
        Parameter for parallel computation.
        
    Returns
    -------
    pred : ndarray
    '''
    pred_list = []
    pred_list = Parallel(n_jobs=n_jobs)(delayed(model.predict)(X)for model in models) # parallel prediction
    for i, _ in enumerate(pred_list): # reshape the prediction to concatenate
        pred_list[i] = pred_list[i].reshape(X.shape[0], 1)
    pred = np.concatenate(pred_list, axis=1)
    return pred


def count_leaf(tree):
    '''
    Count the number of rules.
        
    Parameters
    ----------
    tree : integer, required
        The tree index for extracting rule. It should be smaller than the number of the tree in the random forest.
        
    Returns
    -------
    sum_leaf : int
    '''
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    is_leaf = np.logical_and(children_left<0, children_right<0) # if both the children_left and the children_right are less than 0
    sum_leaf = is_leaf.sum()
    return(sum_leaf)


def extract_rule(tree, n_rule, feature_names=None):
    '''
    Get a rule from the specific tree.
        
    Parameters
    ----------
    tree : sklearn.tree, required
        The tree model for extracting rules.
        
    n_rule : integer, required
        The rule index for extracting rules.

    feature_names : list, optional
        
    Returns
    -------
    rule[is_leaf][n_rule] : string
    depth : array
    '''
    # Extracting rules from the tree of the specific layer
    t = tree.tree_
    children_left = t.children_left
    children_right = t.children_right
    feature = t.feature
    threshold = t.threshold
    n_nodes = t.node_count
    is_leaf = np.logical_and(children_left<0, children_right<0)
    sum_leaf = is_leaf.sum()
    print("num of rules: ", sum_leaf)

    node_id = 0
    rule_count = 0
    if feature_names:
        temp_rule = "{}".format(feature_names[feature[node_id]])# the feature for the root node        
    else:
        temp_rule = "X[,%d]"%(feature[node_id])# the feature for the root node        
    rule = np.zeros(shape=n_nodes, dtype="U10000") # the rule string array for each node
    depth = np.zeros(shape=n_nodes, dtype="int64")
    que = [] # queue for waiting node_id
    current_depth = 0
    
    # Traversal the tree
    while(rule_count < sum_leaf):
        left = children_left[node_id]
        right = children_right[node_id]
        current_depth = depth[node_id]
        
        # If the node has left or right child node, adding the threshold and put the temp rule into the rule[node] array.
        if(left > 0):
            temp_left = temp_rule + "<=%.3f"%(threshold[node_id])
            rule[left] = temp_left # Store the rule of the left child node to the array
            que.append(left) # Append the node_id of the left child to the que
            depth[left] = current_depth + 1
        if(right > 0):
            temp_right = temp_rule + ">%.3f"%(threshold[node_id])
            rule[right] = temp_right
            que.append(right)
            depth[right] = current_depth + 1
        # If the node has no child node, it is a leaf node.
        if(left < 0 and right < 0):
            rule[node_id] = temp_rule
            rule_count += 1
                
        if(rule_count >= sum_leaf):
            break
        
        # Get the next node_id to check
        node_id = que.pop(0)
        # If the next node has feature
        if(feature[node_id] >= 0):
            if feature_names:
                temp_rule = rule[node_id] + " & {}".format(feature_names[feature[node_id]])
            else:
                temp_rule = rule[node_id] + " & X[,%d]"%feature[node_id] # add AND logic and the feature
        else:
            temp_rule = rule[node_id] # leaf node

    return (rule[is_leaf][n_rule], depth)# get the leaf nodes and the rule




def return_feature_imp(models, autoencoder):
    imp = np.concatenate(list(map(lambda m: m.feature_importances_.reshape(-1, 1), models)), axis=1) # 串連所有RF模型的特徵重要性

    # 得到 AE 權重
    w0 = autoencoder.decoder.get_weights()[0]
    b0 = autoencoder.decoder.get_weights()[1]
    w1 = autoencoder.decoder.get_weights()[2]
    b1 = autoencoder.decoder.get_weights()[3]

    # 計算加權後的特徵重要性
    out = (imp.dot(w0)).dot(w1)
    return out
