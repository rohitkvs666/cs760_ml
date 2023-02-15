import numpy as np
from math import log2
from treelib import Tree

def read_data(filename):
    samples = [];
    with open(filename) as fp:
        for line in fp:
            samples.append([float(x) for x in line.split()]);
    return np.array(samples)

def computeEntropy(labels):
    entropy = 0;
    unique, counts = np.unique(labels, return_counts = True);
    for item in counts:
        prob = item/len(labels);
        entropy -= prob*log2(prob);
    return entropy

class decisionTree:
    total_nodes = 0
    all_nodes = []

    def __init__(self, attributes, labels):
        self.report = []
        self.gain_ratio = 0.0
        self.split_condition = "leaf"
        self.node_id = -1
        self.isleaf = 0
        self.children = [None, None]
        self.node_id = decisionTree.total_nodes
        self.parent = None
        count0 = len(np.where(labels[:]==0)[0])
        count1 = len(np.where(labels[:]==1)[0])
        self.majority = self.split_condition = int(count1 >= count0)

        decisionTree.all_nodes.append(self)

        decisionTree.total_nodes += 1; #take care of leaf increments.
        if True: #decisionTree.total_nodes < 6:
            self.build_tree(attributes, labels)

    def __str__(self):
        return "node_id : %d, split_attr: %d, gain_ratio : %f \n"  \
            "is_leaf : %d, parent_id : %s, num_instances : %d \n" \
            % (self.node_id, self.split_attr, self.gain_ratio, self.isleaf,
               (-1 if (self.parent == None) else self.parent.node_id),
               self.num_instances)

    def build_tree(self, attributes, labels):
        Hd_node = computeEntropy(labels)
        self.num_instances = count = len(labels)

        if not count or not Hd_node:
            self.isleaf = 1
            return

        attr_types = np.arange(attributes.shape[1])
        best_attr = -1
        best_gain_ratio = float('-inf')
        best_thresh = float('-inf')

        local_report = []
        for attr in attr_types:
            for thresh in np.unique(attributes[:, attr]):
                ind_child1 = np.where(attributes[:, attr] >= thresh)[0]
                ind_child2 = np.where(attributes[:, attr] < thresh)[0]
                Hd_c1 = computeEntropy(labels[ind_child1])
                Hd_c2 = computeEntropy(labels[ind_child2])
                p_c1 = len(ind_child1)/count
                p_c2 = len(ind_child2)/count

                Hd_given_attr_split = (p_c1 * Hd_c1) + (p_c2 * Hd_c2)

                info_gain = Hd_node - Hd_given_attr_split

                intrinsic_info = computeEntropy(attributes[:,attr]>=thresh)
                #if not intrinsic_info:
                #    continue

                intrinsic_info
                gain_ratio = info_gain / intrinsic_info

                local_report.append([attr, thresh, p_c1, p_c2, gain_ratio, intrinsic_info, info_gain])

                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_attr = attr
                    best_thresh = thresh

        self.c = best_thresh
        self.split_attr = best_attr
        self.gain_ratio = best_gain_ratio
        self.report = np.array(local_report)

        if not best_gain_ratio:
            self.isleaf = 1
            return

        split_cond = "X2" if best_attr else "X1"
        split_cond += ">=%f" %best_thresh
        self.split_condition = split_cond

        #Call children nodes
        ind_child1 = np.where(attributes[:, best_attr] >= best_thresh)[0]
        ind_child2 = np.where(attributes[:, best_attr] < best_thresh)[0]
        #split_indices = [ind_child1, ind_child2]
        #for ind_child in split_indices:
        self.children[0] = (decisionTree(attributes[ind_child1], labels[ind_child1]))
        self.children[1] = (decisionTree(attributes[ind_child2], labels[ind_child2]))
        self.children[0].parent = self
        self.children[1].parent = self

def main():
    global tree_root
    print("Test Codeee")
    samples = read_data("./D2.txt")
    attributes = samples[:, :-1]
    labels = samples[:,-1] #last column is labels

    tree_root = decisionTree(attributes, labels)

    #Tree Generation
    full_tree = Tree()
    for node in decisionTree.all_nodes:
        if node.parent == None:
            full_tree.create_node(str(node.node_id), node.node_id, data = node)
        else:
            full_tree.create_node(str(node.node_id), node.node_id,
                                  parent = node.parent.node_id, data = node)

    print("\n")
    full_tree.show(line_type="ascii-em", data_property="split_condition")
    full_tree.show(line_type="ascii-em", data_property="num_instances")

if __name__ == "__main__":
    main()