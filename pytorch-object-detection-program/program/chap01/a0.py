#!/usr/bin/python
# -*- coding: sjis -*-

import numpy as np

def f(x):  #  �֐��̒�`
    return (x[0] -2 * x[1] -1)**2 + (x[1] * x[2] -1)**2 + 1

def f_grad(x):  #  ���֐��̒�`
    g1 = 2 * (x[0] -2 * x[1] -1)
    g2 = -4 * (x[0] -2 * x[1] -1) + 2 * x[2] * (x[1] * x[2] -1)
    g3 = 2 * x[1] * (x[1] * x[2] -1)
    return np.array([g1, g2, g3])

x = np.array([1.0, 2.0, 3.0])  #  �����l
for i in range(50):  # 50 ��̌J��Ԃ��C���̉񐔂͓K��
    x = x - 0.1 * f_grad(x)     #  �ŋ}�~���@
    print("x = ",x,", f = ",f(x))
