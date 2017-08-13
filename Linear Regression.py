# -*- coding: utf-8 -*-
import numpy as np
from load_data import load_data

def sigmoid(inX):
    '''
    sigmoid 函数
    '''
    return 1.0/(1+exp(-inX))

def training(data, labels, alpha, max_iters):
    '''
    采用梯度下降法对参数进行更新
    # alpha： 步长
    # max_iters： 最大迭代次数
    '''    
    
    data_mat = mat(data)
    label_mat = mat(labels)
    n_row, n_col = shape(data_mat)
    weights = np.ones((n_col, 1))
    for i in xrange(max_iters):
        outputs = sigmoid(data_mat*weights)
        error = labels - outputs
        weights = weights + alpha*data_mat.transpose()*error
    return weights
    
def predict(data, weights):
    '''
    对测试集进行预测
    '''
    data_mat = mat(data)
    n_row, n_col = shape(data_mat)
    predict = sigmoid(data_mat*weights)
    pre_labels = np.zeros(n_row)
    for i in xrange(n_row):
        if predict[i] >= 0.5:
            pre_labels[i] = 1
        else:
            pre_labels[i] = 0
    return pre_labels

def test_error(labels, output):
    error = 0.0
    labels = mat(labels)
    n_row = len(labels)
    error = np.sum((labels-output)**2)/labels
    return error
def logistic_regression(train_name, alpha, max_iters, test_name):
    '''
    logistic regression 函数的实现流程
    alpha: 参数学习率
    max_iters: 最大迭代次数
    train_name: 训练数据的的路径文件名
    test_name：测试数据的路径名
    '''
    train_data, train_labels = load_data(train_name)
    test_data, test_labels = load_data(test_name)
    weights = training(train_data, train_labels, alpha, max_iters)
    output = predict(weights, test_data)
    test_error(test_labels, output)
if __name__ == "__main__":
    train_name = 'train_data.txt'
    test_name = 'test_data.txt'
    alpha = 0.01
    max_iters = 500
    logistic_regression(train_name, test_name)
