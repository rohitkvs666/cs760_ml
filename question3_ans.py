#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 08:47:32 2022

@author: charls
"""

from math import pi
import numpy as np
import matplotlib.pyplot as plt

def lagrange_fit(x, x_list, y_list):
    res = []
    for item in x:
        lj = []
        for xj in x_list:
            x_no_j = np.delete(x_list, np.where(x_list==xj))
            num = item - x_no_j
            den = xj - x_no_j
            lj.append(np.product(num)/np.product(den))
        res.append(np.sum(lj*y_list))
    return res

def gauss_set(n, mu, var):
    gauss_x = np.random.normal(mu, var, n)
    gauss_x = gauss_x[np.nonzero(gauss_x>=0)]
    gauss_x = gauss_x[np.nonzero(gauss_x<=4*pi)]
    gauss_y = np.sin(gauss_x)
    return [gauss_x, gauss_y]

def uni_set(n):
    x = np.random.uniform(0, 4*pi, n)
    return [x, np.sin(x)]

n = 100
mu = 0
var = pi/2

[uni_x, uni_y] = uni_set(n)
[gauss_x, gauss_y] = gauss_set(n, mu, var)

## Train Error
diff_uni = lagrange_fit(uni_x, uni_x, uni_y) - uni_y
diff_gauss = lagrange_fit(gauss_x, gauss_x, gauss_y) - gauss_y
gauss_error = np.sqrt(np.square(diff_gauss).mean())
uni_error = np.sqrt(np.square(diff_uni).mean())

print("train uni error =", uni_error)
print("train gauss error =", gauss_error)

## Test Error
[test_uni_x, test_uni_y] = uni_set(n)
[test_gauss_x, test_gauss_y] = gauss_set(n, mu, var)

diff_uni = lagrange_fit(test_uni_x, uni_x, uni_y) - test_uni_y
uni_error = np.sqrt(np.square(diff_uni).mean())
print("test uni error ", uni_error)

var = [pi/6, pi/4, pi/2]
for v in var:
    [gauss_x, gauss_y] = gauss_set(n, mu, v)
    diff_gauss = lagrange_fit(test_gauss_x, gauss_x, gauss_y) - test_gauss_y
    gauss_error = np.sqrt(np.square(diff_gauss).mean())

    print("test gauss error =", gauss_error)