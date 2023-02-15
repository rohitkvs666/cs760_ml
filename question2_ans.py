#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 01:45:23 2022

@author: charls
"""
from sklearn import tree
from build_tree import read_data
import numpy as np
import matplotlib.pyplot as plt
import copy

samples = read_data("./Dbig.txt")

rand_list = np.random.RandomState(seed=3).permutation(10000)

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
clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=0)

for count, t_list in enumerate(training_list):
    attributes = t_list[:, :-1]
    labels = t_list[:,-1]

    clf = clf.fit(attributes, labels)
    plt.figure()
    tree.plot_tree(clf)
    plt.show()

    dtree.append(copy.deepcopy(clf))
    dtree_node_count.append(clf.tree_.node_count)

    color = np.where(labels, 'r', 'b')
    plt.figure()
    plt.scatter(attributes[:, 0], attributes[:, 1], color = color)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Scatter plot D%d" %training_n[count])
    plt.show()

error = []
results = []
for tr in dtree:
    res = tr.predict(Dtest[:, :-1])
    results.append(res)
    error.append(len(np.where(res.astype(int)!=Dtest[:,2])[0]))

conc = [training_n[:], dtree_node_count[:], error[:]]
for count, item in enumerate(error):
    print(training_n[count], dtree_node_count[count], error[count])
print(conc)

print(training_n, dtree_node_count, error)
#plt.figure()
plt.plot(training_n, error)
plt.xlabel("n")
plt.ylabel("error")
plt.title("Training Set Size vs Error")