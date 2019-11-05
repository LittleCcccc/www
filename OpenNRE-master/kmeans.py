import sklearn
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.spatial.distance as dist

from extract_feature import BertVector
import json
def train(X,show_label = False):
    num_clusters = 2
    k_means = sklearn.cluster.KMeans(
        n_clusters=num_clusters,#簇的个数
         init='k-means++',#初始簇中心的获取方法
        n_init=10,
        max_iter=300,# 最大迭代次数
        tol=0.0001,#容忍度，即kmeans运行准则收敛的条件
        precompute_distances='auto',
        verbose=0,
        random_state=None,# 随机生成簇中心的状态条件
        copy_x=True,
        n_jobs=1,
        algorithm='auto'
        )
    result = k_means.fit(X)
    print (result)
    print(k_means.cluster_centers_)
    print(k_means.labels_)
    print(k_means.inertia_)

file = open("./data/iddict2.json",'r',encoding='utf-8')
iddict = json.load(file)

def eculidis(vector1,vector2):
    op2 = np.linalg.norm(vector1 - vector2)
    return op2
def manhattandis(vector1,vector2):
    op4 = np.linalg.norm(vector1 - vector2, ord=1)
    return op4
def chebdis(vector1,vector2):
    op6 = np.linalg.norm(vector1 - vector2, ord=np.inf)
    return op6
def cosdis(vector1,vector2):
    op7 = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
    return op7
def hanmingdis(v1,v2):
    smstr= np.nonzero(v1 - v2)
    sm= np.shape(smstr[0])[0]
    return sm
def Jaccarddis(v1,v2):
    matv = np.array([v1, v2])
    ds=dist.pdist(matv,'jaccard')
    return ds


if __name__ == "__main__":
    list = []
    cnt = 0
    for item in iddict.values():
        list.append(item['sentence'])
        print(item)
        print(cnt)
        cnt += 1
        if cnt==100:
            break
    bert = BertVector()
    v = bert.encode(list)
    v1 = bert.encode(["犯罪以后自动投案，如实供述自己的罪行的，是自首。"])
    #SSE = []  # 存放每次结果的误差平方和
    target = v1[0]
    newslist = []
    newslist2 = []
    newslist3 = []
    newslist4 = []
    newslist5 = []
    newslist6 = []
    cnt=0
    for item in v:
        newslist.append((cnt,list[cnt],eculidis(item,target)))
        newslist2.append((cnt,list[cnt],manhattandis(item,target)))
        newslist3.append((cnt,list[cnt],chebdis(item,target)))
        newslist4.append((cnt,list[cnt],cosdis(item,target)))
        newslist5.append((cnt,list[cnt],hanmingdis(item,target)))
        newslist6.append((cnt,list[cnt],Jaccarddis(item,target)))
        cnt += 1
    newslist.sort(key=lambda x:x[2],reverse=False)
    newslist2.sort(key=lambda x:x[2],reverse=False)
    newslist3.sort(key=lambda x:x[2],reverse=False)
    newslist4.sort(key=lambda x:x[2],reverse=False)
    newslist5.sort(key=lambda x:x[2],reverse=False)
    newslist6.sort(key=lambda x:x[2],reverse=False)
    print (newslist)
    print (newslist2)
    print (newslist3)
    print (newslist4)
    print (newslist5)
    print (newslist6)

    #estimator = KMeans(n_clusters=500)  # 构造聚类器
    #rst = estimator.fit(v)
    #SSE.append(estimator.inertia_)
    #print(k)
    #print(estimator.inertia_)
    #X = range(1, 50)
    #plt.xlabel('k')
    #plt.ylabel('SSE')
    #plt.plot(X, SSE, 'o-')
    #plt.show()


    '''
    rstdict = {}
    for i in range (500):
        rstdict[str(i)]=[]

    #print(str(v))
    #train(v)
    print(rst.labels_)
    cnt = 0
    for i in rst.labels_:
        rstdict[str(i)].append(iddict[str(cnt)]['sentence'])
        cnt += 1
    print(rstdict)
    file = open("./data/rstdict500.json",'w',encoding='utf-8')
    json.dump(rstdict,file,ensure_ascii=False)
    '''


