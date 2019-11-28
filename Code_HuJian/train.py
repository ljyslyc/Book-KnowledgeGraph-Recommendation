import data_process as dpc
import numpy as np
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import math

WORDS, Two_Word = dpc.get_TwoWords()

# 训练标签获取
with open('en.txt') as f:
    flag = f.read().splitlines()
    train_flag = dict()
    cnt = 0
    for line in flag:
        line = line.split(':::')
        if(line[1] == 'male') :
            train_flag[line[0]] = 1
        else :
            train_flag[line[0]] = 0
        cnt += 1
        # print(train_flag[line[0]])
    print('训练标签获得...')
    print(train_flag)
vector = 100
# 向量数量
cnt = 0
mp = dict()
for item in Two_Word:
    mp[item[0]] = cnt
    cnt += 1
    if(cnt >= vector):
        break

str = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~#￥%……&*（）]+"

# 训练集X向量获得
with open('train_txt.csv') as f:
    txt = f.read().splitlines()
    X = np.zeros((len(txt)//100, vector), dtype=np.int)
    Y = np.zeros(len(txt)//100, dtype=np.int)
    cnt = 0
    j = 0
    for line in txt:
        line = line.split(',')
        words = line[2].split(' ')
        anthur = line[1]
        Y[cnt] = train_flag[anthur]
        first = re.sub(str, "", words[0])
        i = 0

        for word in words:
            if i==0:
                i += 1
                continue
            word = re.sub(str, "", word)
            if(first + " " + word in mp.keys()):
                X[cnt][mp[first + " " + word]] += 1
        j += 1
        if(j == 100):
            j = 0
            cnt += 1
    print('训练向量获得...')
    print(X)

# 测试标签获取
with open('en_test.txt', 'r') as f:
    flag_text = f.read().splitlines()
    test_flag = dict()
    cnt = 0
    for line in flag_text:
        line = line.split(':::')
        if (line[1] == 'male'):
            test_flag[line[0]] = 1
        else:
            test_flag[line[0]] = 0
        cnt += 1
    print('测试标签获得...')
    print(test_flag)

#测试向量获取
with open('test_txt.csv') as f:
    txt = f.read().splitlines()
    X_test = np.zeros((len(txt)//100, vector), dtype=np.int)
    Y_test = np.zeros(len(txt)//100, dtype=np.int)
    cnt = 0
    j = 0
    for line in txt:
        line = line.split(',')
        words = line[2].split(' ')
        anthur = line[1]
        first = re.sub(str, "", words[0])
        i = 0

        Y_test[cnt] = test_flag[anthur]
        for word in words:
            if i==0:
                i += 1
                continue
            word = re.sub(str, "", word)
            if(first + " " + word in mp.keys()):
                X_test[cnt][mp[first + " " + word]] += 1
        j += 1
        if(j == 100):
            cnt += 1
            j = 0
    print('训练向量获得...')
    print(X_test)

# 训练--结果获取

print('开始训练...')
# 朴素贝叶斯分类
clf = GaussianNB() #高斯朴素贝叶斯
clf2 = BernoulliNB() # 贝努利
clf3 = MultinomialNB() # 多项式

clf.fit(X, Y)
clf2.fit(X, Y)
clf3.fit(X, Y)
max_len = len(Y_test)
# print(max_len)
acc = 0
acc2 = 0
acc3 = 0
for i in range(max_len):
    pre = clf.predict(X_test[i, :].reshape(1, vector))
    if(pre == Y_test[i]):
        acc += 1
    pre2 = clf2.predict (X_test[i, :].reshape (1, vector))
    if (pre2 == Y_test[i]):
        acc2 += 1
    pre3 = clf3.predict (X_test[i, :].reshape (1, vector))
    if (pre3 == Y_test[i]):
        acc3 += 1
    else:
        print("{0}: 预测：{1} 实际：{2}".format(i, pre3, Y_test[i]))
acc = acc/(max_len)
print('Gauss acc :{0}'.format(acc))
acc2 = acc2/(max_len)
print('Bernou acc :{0}'.format(acc2))
acc3 = acc3/(max_len)
print('Multi acc :{0}'.format(acc3))


# SVR分类器

# svr = SVC(C=100, gamma=0.1)
# svr.fit(X, Y)
# # print(max_len)
# acc = 0
# max_len = len(Y_test)
# for i in range(max_len):
#     pre = svr.predict(X_test[i, :].reshape(1, vector))
#     print(pre)
#     if(math.fabs(pre-Y_test[i]) < 0.5):
#         acc += 1
# acc = acc/(max_len)
# print('SVR acc :{0}'.format(acc))


