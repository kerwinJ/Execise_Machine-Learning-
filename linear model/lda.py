'''
《机器学习》线性判别分析实现
主要代码参考博客： http://blog.csdn.net/wzmsltw/article/details/50994335
'''

# -*- coding: utf-8 -*-

from numpy import *
import numpy as np
import pandas as pd
import matplot.pyplot as plt

df = pd.read_csv('watermelon_3a.csv')

def training():
    df1 = df[df.label == 1]
    df2 = df[df.label == 0]
    X1 = df1.values[:, 1:3]
    X0 = df2.values[:, 1:3]
    mean1 = array([mean(X1[:, 0]), mean(X1[:,1])])
    mean0 = array([mean(X0[:, 0]), mean(X0[:,1])])
    num0 = shape(X0)[0]
    num1 = shape(X1)[0]
    sw = zeros(shape=(2,2))
    for i in xrange(num0):
        xsmean = mat(X0[i,:]-mean0)
        sw += xsmean.transpose()*xsmean
    for i in xrange(num1):
        xsmean = mat(X1[i,:]-mean1)
        sw += xsmean.transpose()*xsmean
    w = (mean0 - mean1)* (mat(sw).I)
    return w

def plot(data, weights):
    data_mat = array(df['density', 'radio_suger'].values[:,:])
    label_mat = mat(df['label'].values[:]).transpose()
    m = shape(data_mat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in xrange(m):
        if label_mat[i] == 1:
            xcord1.append(data_mat[i])
            ycore1.append(label_mat[i])
        else:
            xcord2.append(data_mat[i])
            ycord2.append(label_mat[i])
    plt.figure(1)
    ax = plt.subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='greeen')
    x = arange(-0.2, 0.8, 1)
    y = array((-w[0,0]*x)/w[0,1])
    print shape(x)
    print shape(y)
    plt.sca(ax)
    
    plt.plot(x,y)
    plt.xlabel('density')
    plt.ylabel('radio_suger')
    plt.title('LDA')
    plt.show()
   
def lda():
    weights = training()
    plot()
    
if __name__ == '__main__":
    lda()
