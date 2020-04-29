import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from  sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import joblib
import graphviz

from sklearn.model_selection import cross_val_score

base_dataset_dir = '/home/a/Downloads/NetworkAnalysis/dataset/'
dataset_dir = '/home/a/Downloads/NetworkAnalysis/dataset/data'
out_dir = '/home/a/Downloads/NetworkAnalysis/dataset/'
headers = ['index', 'time_diff', 'scr_ip', 'dst_ip', 'protocol',
           'length', 'content', 'seq', 'ack', 'retrans_flag',
           'wsize', 'rtt', 'uncertain_bytes', 'tcp_flag', 'label']
removed_feature = ['scr_ip', 'dst_ip',  'content', 'retrans_flag']

# 数据预处理
# 读入数据
# 初始化header
header_flag = True
home, dirs, files = list(os.walk(dataset_dir))[0]
for file_name in files:
    #获取lable
    label = file_name[0: file_name.find('-')]
    # 初始化header
    data = pd.read_csv(os.path.join(home, file_name), names=headers)
    data['label'] = label
    # 填充0
    data.fillna(0, inplace=True)
    # 添加属性direction
    # #客户端-> 公网： 0
    data['direction'] = data['dst_ip'].map(lambda ip: 1 if ip == '1.1.1.1' else 0)
    # 转换属性protocol, tcp_flag,one-hot编码
    data['protocol'] = data['protocol'].map({
        'TCP': 0,
        'TLSv1': 1,
        'TLSv1.3': 2,
        'SSLv2': 3
    })
    data['tcp_flag'] = data['tcp_flag'].map({
        '0x00000002': 0,
        '0x00000012': 1,
        '0x00000010': 2,
        '0x00000018': 3,
        '0x00000011': 4,
        '0x00000004': 5
    })
    # 消除属性index, scr_ip, dst_ip, protocol, content, retrans_flag, tcp_flag
    data.drop(removed_feature, axis=1, inplace=True)
    data.to_csv(out_dir + 'data_single.csv', mode='a', index=False, header=header_flag)
    header_flag = False



# 第三阶段，模型训练
# 导入数据，划分训练集和测试集
data = pd.read_csv(out_dir + 'data_single.csv', skipinitialspace=True)
X = data[data.columns.difference(['label'])]
#normalization
# x_normalized = preprocessing.normalize(X, norm='l2')
# min_max_scaler = preprocessing.MinMaxScaler()
# x_train_minmax = min_max_scaler.fit_transform(X)

# scaler = preprocessing.StandardScaler().fit(X)
# X_score = scaler.transform(X)

y = data['label']
feature_names = X.columns
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#################KNN#############################
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, Y_train)
# y_train = knn.predict(X_train)
# y_pred = knn.predict(X_test)
# print(metrics.classification_report(Y_train, y_train))
# print(metrics.classification_report(Y_test, y_pred))

#####################NB#######################
# nb = GaussianNB()
# nb.fit(X_train, Y_train)
# y_train = nb.predict(X_train)
# y_pred = nb.predict(X_test)
# print(metrics.classification_report(Y_train, y_train))
# print(metrics.classification_report(Y_test, y_pred))

###################CART########################
clf = tree.DecisionTreeClassifier(criterion='gini')
clf.fit(X_train, Y_train)
y_train = clf.predict(X_train)
y_pred = clf.predict(X_test)
print(metrics.classification_report(Y_train, y_train))
print(metrics.classification_report(Y_test, y_pred))
# dot_data = tree.export_graphviz(clf, out_file=None,
#                          # feature_names=feature_names,
#                          # class_names=list(range(50)),
#                          filled=True, rounded=True,
#                          special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.render('tree')






################################NO RESULTS##########################

#########################RandomForestClassifier##############
# rfc = RandomForestClassifier()
# rfc.fit(X_train, Y_train)
# joblib.dump(rfc,'rfc.pkl')
# y_pred = rfc.predict(X_test)
# print(metrics.classification_report(Y_test, y_pred))




#######################ADABOOST##################
# ABC = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy'), algorithm="SAMME",
#                         n_estimators=200, learning_rate=0.8)
# ABC.fit(X_train, Y_train)
# y_pred = ABC.predict(X_test)
# print(metrics.classification_report(Y_test, y_pred, target_names=target_names))



# svm
# SVM = svm.SVC(kernel='rbf', gamma=10)
# SVM = svm.SVC()
# SVM.fit(X_train, Y_train)
# y_pred = SVM.predict(X_test)
# print(metrics.classification_report(Y_test, y_pred, target_names=target_names))
