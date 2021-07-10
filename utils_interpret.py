from sklearn.linear_model import Lasso
from sklearn.utils import resample
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import re

def get_num_rule(onehot_encoder): # 從 rule 的 one-hot encoder取得
    '''
    Get the number of rules of each tree.
    
    Parameters
    ----------
    onehot_encoder : scikitlearn.preprocessing.OneHotEncoder, required
        One-hot encoder of the rules.
        
    Returns
    -------
    num_rule : list
    '''
    num_rule = [len(n) for n in onehot_encoder.categories_] #每棵樹包含的規則數量
    return(num_rule)


def return_rule_coef(lm_models, autoencoder):
    '''
    Get the weighted coefficients of each rule from linear models.
    
    Parameters
    ----------
    lm_models : list of LinearRegression
    
    autoencoder : Autoencoder_NoneHid()
        
    Returns
    -------
    weighted_coef : ndarray
    '''
    coefs = np.concatenate(list(map(lambda m: m.coef_.reshape(-1, 1), lm_models)), axis=1) # 串連所有線性模型的係數
    # 得到 AE 權重
    w0 = autoencoder.decoder.get_weights()[0]
    b0 = autoencoder.decoder.get_weights()[1]

    # 計算加權後的特徵重要性
#     weighted_coef = autoencoder.decoder.predict(coefs)
    weighted_coef = coefs.dot(w0) + b0
    return weighted_coef


def get_rule_bound(num_rule):
    '''
    Get the index bound of each rule.
    
    Parameters
    ----------
    num_rule : list
        
    Returns
    -------
    rule_bound : list
    '''
    rule_bound = [num_rule[0]-1] # 第一棵樹的 rule bound 到規則的數量 -1
    for i in range(len(num_rule)-1): # 加上各棵樹的規則的數量變成各棵樹的 rule bound
        rule_bound.append(rule_bound[i] + num_rule[i+1])
    return(rule_bound)


def get_tree_by_index(index, rule_bound): #從 rule index 得到是第幾棵樹
    '''
    Get tree index by rule index.
    
    Parameters
    ----------
    index : int
        The rule index.
    
    rule_bound : list
        
    Returns
    -------
    t_idx : int
    '''
    for bound in rule_bound: # 尋找 rule 所屬的 tree
        if bound >= index: # 如果 index 在 bound 的範圍內，回傳 bound 屬於第幾棵樹
            t_idx = rule_bound.index(bound)
            return(t_idx)
    

def get_model_by_index(index, rule_bound, models): # 從 rule index 得到是第幾個模型
    '''
    Get model index by rule index.
    
    Parameters
    ----------
    index : int
        The rule index.
    
    rule_bound : list
    
    models : scikitlearn.RandomforestRegressors
        
    Returns
    -------
    m_idx : int
    '''
    t_idx = get_tree_by_index(index, rule_bound)
    n_tree = 0 # 各模型的樹的累積數量
    
    for m_idx, m in enumerate(models):
        n_tree += len(models[m_idx].estimators_)
        if t_idx < n_tree:
            return(m_idx)

        
def get_rule_by_index(index, rule_bound, models, feature_names): # 從 rule bound 得到規則
    '''
    Get rule by rule index.
    
    Parameters
    ----------
    index : int
        The rule index.
    
    num_rule : list
    
    models : scikitlearn.RandomforestRegressors
    
    feature_names : list
        List of feature names.
    
    Returns
    -------
    rule_text[rule_order] : str
    '''
    tree_order, rule_order = get_rule_order_in_tree_by_index(index, rule_bound, models)
#     rule_bound = get_rule_bound(num_rule)
#     t_idx = get_tree_by_index(index, rule_bound)
    m_idx = get_model_by_index(index, rule_bound, models)
#     n_tree = 0
#     for n in range(m_idx):
#         n_tree += len(models[n].estimators_)
#     tree_order = t_idx - n_tree # 得到 t_idx 屬於第幾個模型的第幾棵樹
    rule_text = extract_rules_from_tree(models[m_idx].estimators_[tree_order], feature_names=feature_names) # 得到一棵樹的所有文字規則
#     max_rule_idx_pre = 0 if (index <= rule_bound[0]) else rule_bound[t_idx-1] + 1 # 前一棵樹的 rule bound
#     rule_order = index - max_rule_idx_pre # 得到 index 屬於目前樹的第幾條規則
    return(rule_text[rule_order]) 

def get_rule_order_in_tree_by_index(index, rule_bound, models): 
    '''
    Get rule by rule index.
    
    Parameters
    ----------
    index : int
        The rule index.
    
    rule_bound : list
    
    models : scikitlearn.RandomforestRegressors
    
    
    Returns
    -------
    tree_order: int
    rule_order: int
    '''
    t_idx = get_tree_by_index(index, rule_bound)
    m_idx = get_model_by_index(index, rule_bound, models)
    n_tree = 0
    for n in range(m_idx):
        n_tree += len(models[n].estimators_)
    tree_order = t_idx - n_tree # 得到 t_idx 屬於第幾個模型的第幾棵樹

    max_rule_idx_pre = 0 if (index <= rule_bound[0]) else rule_bound[t_idx-1] + 1 # 前一棵樹的 rule bound
    rule_order = index - max_rule_idx_pre # 得到 index 屬於目前樹的第幾條規則
    return(tree_order, rule_order)



def get_rules_by_label(label_idx, rule_bound, weighted_coef, topn, models, feature_names, return_text=False, sort=True):
    '''
    Get associative rules by label index.
    
    Parameters
    ----------
    label_idx : int
        The label index.
    
    weighted_coef : array
    
    topn : int
        The top n associative rule

    feature_names : list
        List of feature names.
    
    Returns
    -------
    rules : str    
    
    coefs, rule_list, topn_rule_idx
    
    '''
    selected_coef = weighted_coef[:, label_idx].copy() # 選擇特定標籤的規則係數

    if sort:
    # 排序從最重要的規則進行萃取，找出前 topn 個重要的規則
        topn_rule_idx = selected_coef.argsort()[::-1][:topn]
        
    else:
        topn_rule_idx = np.arange(selected_coef.shape[0])
    print("search label: '{}'".format(label_idx))
    rule_list = list(map(lambda idx: get_rule_by_index(idx, rule_bound, models, feature_names), topn_rule_idx))
    
    # 回傳文字規則
    if return_text:
        rules = "{:.2f}*({})".format(selected_coef[topn_rule_idx[0]], rule_list[0])
        for i, rule in enumerate(rule_list[1:]):
            rules += " | {:.2f}*({})".format(selected_coef[topn_rule_idx[i]], rule)
        return(rules)
    # 回傳規則係數
    else:
        coefs = [selected_coef[idx] for idx in topn_rule_idx]
        return(coefs, rule_list, topn_rule_idx)

    
def extract_rules_from_tree(tree, feature_names=None):
    '''
    Get associative rules by label index.
    
    Parameters
    ----------
    tree : sklearn.tree
    
    feature_names: list
    
    Returns
    -------
    rules : str    
    
    coefs, rule_list, topn_rule_idx
    
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
#     print("num of rules: ", sum_leaf)

    node_id = 0
    rule_count = 0
    if feature_names is not None:
        temp_rule = "{}".format(feature_names[feature[node_id]])# the feature for the root node
    else:
        temp_rule = "X[,%d]"%(feature[node_id])# the feature for the root node        
    rule = np.zeros(shape=n_nodes, dtype="U10000") # the rule string array for each node
    que = [] # queue for waiting node_id
    
    # Traversal the tree
    while(rule_count < sum_leaf):
        left = children_left[node_id]
        right = children_right[node_id]
        
        # If the node has left or right child node, adding the threshold and put the temp rule into the rule[node] array.
        if(left > 0):
            temp_left = temp_rule + "<=%.3f"%(threshold[node_id])
            rule[left] = temp_left # Store the rule of the left child node to the array
            que.append(left) # Append the node_id of the left child to the que
        if(right > 0):
            temp_right = temp_rule + ">%.3f"%(threshold[node_id])
            rule[right] = temp_right
            que.append(right)
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
            if feature_names is not None:
                temp_rule = rule[node_id] + " & {}".format(feature_names[feature[node_id]])
            else:
                temp_rule = rule[node_id] + " & X[,%d]"%feature[node_id] # add AND logic and the feature
        else:
            temp_rule = rule[node_id] # leaf node

    return rule[is_leaf]# get the leaf nodes and the rule



def compute_feature_coef(label_idx, rule_bound, weighted_coef, models, feature_names, categorical_feature=True):
    '''
    Compute the feature coefficients for a specific label.
    
    Parameters
    ----------
    label_idx : int
        The label index.
        
    num_rule : list
    
    weighted_coef : array
    
    models : list of RandomForestRegressor

    feature_names : list
        List of feature names.
    
    Returns
    -------
    feature_coef : array
    '''
    coef, rules, rule_idx = get_rules_by_label(label_idx, rule_bound, weighted_coef, None, models, None, False, True)
    feature_coef = np.zeros(len(feature_names))

    for i, idx in enumerate(rule_idx): # 針對選定的標籤計算特徵的總係數
        selected_feature = np.array([int(feature.split(",")[1]) for feature in re.findall(",[0-9]+", rules[i])])# 得到規則涉及的特徵
        if categorical_feature:
            categorical = re.findall('[><]', rules[i])
            existent_feature = np.array([0 if (existence=='<') else 1 for existence in categorical])
            feature_coef[selected_feature] += (existent_feature*weighted_coef[rule_idx[i], label_idx])#
        else:
            feature_coef[selected_feature] += weighted_coef[rule_idx[i], label_idx]
    return(feature_coef)


def compute_bootstrap_lasso_coef(X_train, y_train, flter, label_idx, autoencoder,  sample_times=100):
    lasso_coefs = []
    for i in tqdm(range(sample_times)): #抽樣次數
        sub_X_train, sub_y_train = resample(X_train[flter], y_train[flter, label_idx],
                                            n_samples=X_train.shape[0],
                                            random_state=i) # 取出 subgroup 並重複抽樣到原始的樣本數量
        lasso = Lasso(alpha=0.01, max_iter=5000) # 訓練模型
        lasso.fit(sub_X_train.toarray(), sub_y_train)
        
        coefs = lasso.coef_.reshape(1, -1)
        w = autoencoder.decoder.get_weights()[0][label_idx, :].reshape(-1, 1)
        b = autoencoder.decoder.get_weights()[1][label_idx]
        weighted_coef = w.dot(coefs) + b
        
        lasso_coefs.append(weighted_coef) # 得到係數
    boot_coef = np.concatenate(lasso_coefs, axis=0) # 串接係數
    return(boot_coef)



# def plot_confidence_intervals(feature_names, boot_coef, 
#                               topn=5, sorted_idx=None, sorted_coef=None, 
#                               prefix=None, models=None, rule_bound=None):
#     '''
#     X_train:
#     '''
# #     coef_sort_idx = boot_coef.mean(axis=0).argsort()[::-1][:topn] # 抓對特定標籤平均前 5 重要的規則 index
#     coef_sort_idx = sorted_idx
#     topn_bootstrap_coef = boot_coef[:, coef_sort_idx] # 選擇特定的 boot 係數

#     # Compute confidence interval of coefficients
#     intervals = [np.percentile(topn_bootstrap_coef, 2.5, axis=0),
#                  np.percentile(topn_bootstrap_coef, 97.5, axis=0)] # 計算 95% 信賴區間
    
#     # 根據 index 找出特徵名稱
#     fn = list(map(lambda idx: feature_names[idx], coef_sort_idx))
    
#     # Prepare ticks
#     ticks = []
#     prefix = prefix if prefix else ""
#     if models and rule_bound:
#         for idx in coef_sort_idx:
#             t_idx = get_tree_by_index(idx, rule_bound) # 計算樹的 index
#             t_ord, rt_idx = get_rule_order_in_tree_by_index(idx, rule_bound, models)
#             ticks.append("T{}.R{}".format(t_idx, rt_idx)) # For
#     else:
#         ticks = [prefix + str(name) for name in fn]
    
#     # invert y axis
#     ax = plt.gca()
#     ax.invert_yaxis()
# #     ax.set_ylim(ax.get_ylim()[::-1])
    
#     # Plot label
#     plt.ylabel('feature')
#     plt.xlabel('coefficient')

#     plt.vlines(0, ymin=0, ymax=len(intervals[0])-0.5, linestyles='dashed')
#     plt.yticks(range(topn), ticks)
#     plt.plot(sorted_coef, range(topn), '|k')


#     # Plot coefficient interval
#     for idx, (min_int, max_int) in enumerate(zip(intervals[0], intervals[1])):
#         plt.hlines(y=idx, 
#                    xmin=min_int, xmax=max_int, 
# #                    colors=viridis(idx), 
#                    label=fn[idx])
#     plt.show()
    
    
def plot_confidence_intervals(feature_names, boot_coef, 
                              topn=5, sorted_idx=None, sorted_coef=None, 
                              prefix=None, models=None,
                              rule_bound=None, figsize=None,
                              save_fig=None,
                              flt='none', 
                              layout='log odds'):
    '''
    sorted_idx: 係數的規則 index 
    
    sorted_coef: 排序後的 lasso 係數
    
    boot_coef: 進行 bootlasso 後得到的係數
    '''
#     coef_sort_idx = boot_coef.mean(axis=0).argsort()[::-1][:topn] # 抓對特定標籤平均前 5 重要的規則 index
    coef_sort_idx = sorted_idx
    topn_bootstrap_coef = boot_coef[:, coef_sort_idx] # 選擇特定的 boot 係數

    # Compute confidence interval of coefficients
    intervals = [np.percentile(topn_bootstrap_coef, 2.5, axis=0).reshape(sorted_idx.shape[0], 1),
                 np.percentile(topn_bootstrap_coef, 97.5, axis=0).reshape(sorted_idx.shape[0], 1)] # 計算 95% 信賴區間
    
    # Check confidence interval
    intv = np.concatenate(intervals, axis=1)
    is_gr_lowb = np.greater(sorted_coef, intv[:, 0])
    is_ls_upb = np.less_equal(sorted_coef, intv[:, 1])
    is_gr_0 = np.greater(intv[:, 0], np.zeros(shape=sorted_coef.shape))
    is_ls_0 = np.less_equal(intv[:, 1], np.zeros(shape=sorted_coef.shape))
    is_significant = np.greater(intv[:, 0]*intv[:, 1], np.zeros(shape=sorted_coef.shape))
    is_legal = np.logical_and(is_gr_lowb, is_ls_upb)
    
    if flt == "significant" or flt == "abs":
        plot_flter = np.logical_and(is_legal, is_significant)
    elif flt == "positive":
        plot_flter = np.logical_and(is_legal, is_gr_0)
    elif flt == "negative":
        plot_flter = np.logical_and(is_legal, is_ls_0)
    elif flt == "abs_both":
        pos_plot_flter = np.logical_and(is_legal, is_gr_0)
        neg_plot_flter = np.logical_and(is_legal, is_ls_0)
    else:
        plot_flter = np.ones(shape=sorted_coef.shape, dtype='bool')
        
    #
    if flt == "negative":
        coef_sort_idx = coef_sort_idx[plot_flter][-topn:]
        sorted_coef = sorted_coef[plot_flter][-topn:]
        intervals = [intervals[0][plot_flter][-topn:], intervals[1][plot_flter][-topn:]]
    elif flt == "abs":
        coef_sort_idx = np.abs(sorted_coef[plot_flter]).argsort()[::-1]
        sorted_coef = sorted_coef[plot_flter][coef_sort_idx][:topn]
        intervals = [intervals[0][plot_flter][coef_sort_idx][:topn], 
                     intervals[1][plot_flter][coef_sort_idx][:topn]]
        coef_sort_idx = sorted_idx[plot_flter][coef_sort_idx][:topn]
    elif flt == "abs_both":
        neg_coef_sort_idx = coef_sort_idx[neg_plot_flter][-topn:]
        pos_coef_sort_idx = coef_sort_idx[pos_plot_flter][:topn]
        neg_sorted_coef = sorted_coef[neg_plot_flter][-topn:]
        pos_sorted_coef = sorted_coef[pos_plot_flter][:topn]
        coef_sort_idx = np.concatenate([pos_coef_sort_idx, neg_coef_sort_idx])
        sorted_coef = np.concatenate([pos_sorted_coef, neg_sorted_coef])
        i0 = np.concatenate([intervals[0][pos_plot_flter][:topn], intervals[0][neg_plot_flter][-topn:]])
        i1 = np.concatenate([intervals[1][pos_plot_flter][:topn], intervals[1][neg_plot_flter][-topn:]])
        intervals = [i0, i1]
    else:
        coef_sort_idx = coef_sort_idx[plot_flter][:topn]
        sorted_coef = sorted_coef[plot_flter][:topn]
        intervals = [intervals[0][plot_flter][:topn], intervals[1][plot_flter][:topn]]
    
    # 根據 index 找出特徵名稱
    fn = list(map(lambda idx: feature_names[idx], coef_sort_idx))
    
    # Prepare ticks
    ticks = []
    prefix = prefix if prefix else ""
    if models and rule_bound:
        for idx in coef_sort_idx:
            t_idx = get_tree_by_index(idx, rule_bound) # 計算樹的 index
            t_ord, rt_idx = get_rule_order_in_tree_by_index(idx, rule_bound, models)
            ticks.append("T{}.R{}".format(t_idx, rt_idx)) # For
    else:
        ticks = [prefix + str(name) for name in fn]
    
    if figsize:
        fig = plt.figure(figsize=figsize)
    
    # Invert y axis
    if flt != "negative":
        ax = plt.gca()
        ax.invert_yaxis()
#         ax.set_ylim(ax.get_ylim()[::-1])

    # Set font size
    FONT_SIZE = 16
    plt.rc('font', size=FONT_SIZE) 
    
    # Plot
    if layout=='odds':
        plt.ylabel('', fontsize=FONT_SIZE)
        plt.xlabel('Odds Ratio', fontsize=FONT_SIZE)
        plt.grid(True)

        plt.vlines(1, ymin=0-0.5, ymax=len(intervals[0])-0.5, linestyles='dashed')
        plt.yticks(range(sorted_coef.shape[0]), ticks, fontsize=FONT_SIZE)
        plt.plot(np.exp(sorted_coef), range(sorted_coef.shape[0]), 'ok')

        # Print values for points
        for i, v in enumerate(np.exp(sorted_coef)):
            plt.annotate(str("{:.3f}".format(v)), xy=(v,i), xytext=(-18, 7), textcoords='offset points')

        # Plot coefficient interval

        for idx, (min_int, max_int) in enumerate(zip(intervals[0], intervals[1])):
            plt.plot(np.exp(min_int), idx, "|k")
            plt.plot(np.exp(max_int), idx, "|k")
            plt.hlines(y=idx, 
                       xmin=np.exp(min_int),
                       xmax=np.exp(max_int), 
                       label=fn[idx])
        
    else:
        plt.ylabel('', fontsize=FONT_SIZE)
        plt.xlabel('Coefficient', fontsize=FONT_SIZE)
        plt.grid(True)

        plt.vlines(0, ymin=0-0.5, ymax=len(intervals[0])-0.5, linestyles='dashed')
        plt.yticks(range(sorted_coef.shape[0]), ticks, fontsize=FONT_SIZE)
        plt.plot(sorted_coef, range(sorted_coef.shape[0]), 'ok')

        # Print values for points
        for i, v in enumerate(sorted_coef):
            plt.annotate(str("{:.3f}".format(v)), xy=(v,i), xytext=(-25, 7), textcoords='offset points')

        # Plot coefficient interval

        for idx, (min_int, max_int) in enumerate(zip(intervals[0], intervals[1])):
            plt.plot(min_int, idx, "|k")
            plt.plot(max_int, idx, "|k")
            plt.hlines(y=idx, 
                       xmin=min_int,
                       xmax=max_int, 
                       label=fn[idx])
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
#     plt.show()