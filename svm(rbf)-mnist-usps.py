# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 18:28:42 2018

@author: WenDong Zheng
"""

import scipy.io as sio
from sklearn.cross_validation import train_test_split  # 训练测试数据分割   
from sklearn.svm import LinearSVC  
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score  
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt
import time
from sklearn import svm

mnist_train = sio.loadmat('mnist_train.mat')
mnist_train_labels = sio.loadmat('mnist_train_labels.mat')
traindataset1 = mnist_train['mnist_train']
trainlabels1 = mnist_train_labels['mnist_train_labels']

usps_train = sio.loadmat('usps_train.mat')
usps_train_labels = sio.loadmat('usps_train_labels.mat')
traindataset = usps_train['usps_train']
trainlabels = usps_train_labels['usps_train_labels']

#usps dataset    
# 分割数据  
X_train, X_test, Y_train, Y_test = train_test_split(traindataset, trainlabels, test_size=0.25, random_state=33)#训练集:测试集=3:1    
train_start_time = time.time()
#高斯核SVM  
rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.fit(X_train, Y_train) 
train_end_time = time.time() 
test_start_time = time.time()
Y_predict = rbf_svc.predict(X_test) 
test_end_time = time.time()
accuracy = accuracy_score(Y_test,Y_predict)
print('---------------SVM(RBF kernel)------------')
print('Accuracy on the usps dataset: ' , accuracy) 
target_names = ['0', '1', '2','3','4','5','6','7','8','9']
print('usps recognition')
print (classification_report(Y_test, Y_predict, target_names=target_names))

#mnist dataset
# 分割数据  
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(traindataset1, trainlabels1, test_size=0.25, random_state=33)#训练集:测试集=3:1    
train_start_time1 = time.time()
#高斯核核SVM 
rbf_svc = svm.SVC(kernel='rbf') 
rbf_svc.fit(X_train1, Y_train1) 
train_end_time1 = time.time() 
test_start_time1 = time.time() 
Y_predict1 = rbf_svc.predict(X_test1) 
test_end_time1 = time.time()
accuracy1 = accuracy_score(Y_test1,Y_predict1)
print('Accuracy on the mnist dataset: ' , accuracy1) 
target_names1 = ['0', '1', '2','3','4','5','6','7','8','9']
print('mnist recognition')
print (classification_report(Y_test1, Y_predict1, target_names=target_names1))
 
#sum time
print ("mnist train_time used: %.2f(seconds)" % (train_end_time1 - train_start_time1))
print ("mnist test_time used: %.2f(seconds)" % (test_end_time1 - test_start_time1))
print ("usps train_time used: %.2f(seconds)" % (train_end_time - train_start_time))
print ("usps test_time used: %.2f(seconds)" % (test_end_time - test_start_time))