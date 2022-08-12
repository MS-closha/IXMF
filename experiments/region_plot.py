import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02
markers = "ox^"

# Load data
iris = load_iris()

for pairidx, pair in enumerate([[0, 1], ]):
    # We only take the two corresponding features
    X = iris.data[:, pair]
    y = iris.target

    # Train
    clf = DecisionTreeClassifier(max_depth=2).fit(X, y)

    # Plot the decision boundary
#     plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    FONT_SIZE = 20
    plt.rc('font', size=FONT_SIZE) 
    
    x_low = 4
    x_upp = 8
    y_low = 1.5
    y_upp = 4.5
#     for f, v in zip(clf.tree_.feature, clf.tree_.threshold):
#         if f == 0:
#             plt.plot([v, v], [start_point, y_pre], "k-")
#         elif f == 1:
#             plt.plot([start_point, x_pre], [v,v], "k-")
#     plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
    f, v = (clf.tree_.feature, clf.tree_.threshold)

    plt.xlim([4, 8])
    plt.ylim([1.5, 4.5])
    plt.plot([v[0], v[0]], [y_low, y_upp], "k-")
    plt.plot([x_low, v[0]], [v[1], v[1]], "k-")
    plt.plot([v[2], v[2]], [y_low, v[1]], "k-")
    plt.plot([v[4], v[4]], [y_low, y_upp], "k-")
#     plt.plot([v[5], v[5]], [v[1], y_upp], "k-")
#     plt.plot([v[8], v[8]], [y_low, y_upp], "k-")
#     plt.plot([v[0], v[8]], [v[9], v[9]], "k-")
#     plt.plot([v[12], v[12]], [y_low, y_upp], "k-")
    
#     plt.plot([v[2], v[2]], [start_point, y_pre], "k-")
#     plt.plot([v[0], v[2]], [v[3], v[3]], "k-")
#     plt.plot([v[6], v[6]], [start_point, y_pre], "k-")
    
    
#     cs = plt.contour(xx, yy, Z, origin='lower', colors=None, cmap='gray')

    plt.xlabel(iris.feature_names[pair[0]], fontsize=FONT_SIZE)
    plt.ylabel(iris.feature_names[pair[1]], fontsize=FONT_SIZE)

    # Plot the training points
    for i, color, marker in zip(range(n_classes), plot_colors, markers):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1],
                    s=FONT_SIZE*5,
                    c="black",
                    marker=marker, label=iris.target_names[i],
#                     cmap=plt.cm.RdYlBu,
                    edgecolor='black')

# plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
# plt.axis("tight")
plt.savefig("region_plot3.svg", bbox_inches='tight')



plt.figure(figsize=[18, 9])
plot_tree(clf, 
          feature_names=[iris.feature_names[pair[0]], iris.feature_names[pair[1]]],
          proportion=True, impurity=False,
          fontsize=21, 
          class_names=["setosa", "versicolor", "virginica"], 
          label='root', 
          precision=2)
# plt.savefig("tree_structure2.png", bbox_inches='tight')

