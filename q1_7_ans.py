from build_tree import decisionTree, read_data
from treelib import Tree
import numpy as np
import matplotlib.pyplot as plt

samples = read_data("./Dbig.txt")

rand_list = np.random.RandomState(seed=2).permutation(10000)

D8192 = samples[rand_list[:8192]]
D2048 = samples[rand_list[:2048]]
D512 = samples[rand_list[:512]]
D128 = samples[rand_list[:128]]
D32 = samples[rand_list[:32]]

Dtest = samples[rand_list[8192:]]

training_n = [32, 128, 512, 2048, 8192]
training_list = [D32, D128, D512, D2048, D8192]

dtree = []
dtree_node_count = []

for count, t_list in enumerate(training_list):
    attributes = t_list[:, :-1]
    labels = t_list[:,-1]

    decisionTree.total_nodes=0
    decisionTree.all_nodes=[]
    dtree.append(decisionTree(attributes, labels))
    dtree_node_count.append(decisionTree.total_nodes)

    color = np.where(labels, 'y', 'g')
    plt.figure()
    plt.scatter(attributes[:, 0], attributes[:, 1], color = color)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Scatter plot D%d" %training_n[count])
    plt.show()

def infer(test_tree, test_data):
    result = []
    for sample in test_data:
        node = test_tree
        while not node.isleaf:
            node = node.children[0] if (sample[node.split_attr]) < node.c else node.children[1]
        result.append([node.majority==sample[2]])
    return np.array(result)

error = []
for count, item in enumerate(dtree):
    res = infer(item, Dtest)
    error.append(len(np.where(res.astype(int)==0)[0]))

print(training_n, dtree_node_count, error)
plt.plot(training_n, error)
plt.xlabel("n")
plt.ylabel("error")
plt.title("Training Set Size vs Error")
plt.show()
