# -*- coding: utf-8 -*-
from numpy import unique
from numpy import where
import numpy as np
from sklearn.datasets import make_classification,make_blobs

from sklearn.cluster import AffinityPropagation, AgglomerativeClustering,Birch,DBSCAN,KMeans,MiniBatchKMeans,MeanShift,OPTICS,SpectralClustering
from sklearn.mixture import GaussianMixture

from matplotlib import pyplot

from sklearn.decomposition import PCA

#markedFileDir = 'C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/新标签方法切滤/CutWithFilteringFeatureFile_F1score.csv'
markedFileDir = 'C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/新标签方法切滤/CutWithFilteringFeatureFile_Precision.csv'
#markedFileDir = 'C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/新标签方法切滤/CutWithFilteringFeatureFile_Recall.csv'

#主成分分析，数据降维函数开始（这里面的comp就是用来指定降维之后数据的维度）
def getPCAData(data,comp):
    pcaClf = PCA(n_components=comp, whiten=True)
    pcaClf.fit(data)
    data_PCA = pcaClf.transform(data) # 用来降低维度
    return data_PCA
#主成分分析，数据降维函数结束

def spectralClustering(feature_arr):
    X = getPCAData(np.array(feature_arr), 2)         #降维后特征维度为2，这个和下面的pyplot.scatter(X[row_ix, 0], X[row_ix, 1])是对应的，如果为3的话就可以写成pyplot.scatter(X[row_ix, 0], X[row_ix, 1], X[row_ix, 2])
    model = SpectralClustering(n_clusters=6)         #聚类中心个数为6
    model.fit(X)
    yhat = model.fit_predict(X)
    clusters = unique(yhat)
    clusters_index_arr = []
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        clusters_index_arr.append(row_ix)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    #pyplot.show()
    return clusters_index_arr


def bagFeatures(oneBag):
    instanceFeatureNum = 14
    oneBag_arr = oneBag.split(',')
    oneBag_arr.pop()     #去最后一列的标签值
    feature_arr = []        #feature_arr是原始的包里面各个示例特征，一起放在一个二维数组里了
    oneInstance_feature_arr = []
    i = 0
    #print(oneBag_arr)
    for oneBag_arr_element in oneBag_arr:
        i = i + 1
        oneInstance_feature_arr.append(float(oneBag_arr_element))
        if i == instanceFeatureNum:
            feature_arr.append(oneInstance_feature_arr)
            i = 0
            oneInstance_feature_arr = []
    if len(feature_arr) < 6:
        for i in range(10000000000):
            feature_arr = feature_arr + feature_arr
            if len(feature_arr) >= 6:
                break
    #print(len(feature_arr))
    clusters_index_arr = spectralClustering(feature_arr)
    #return feature_arr, clusters_index_arr
    bag_clusters_feature_arr = []        #bag_clusters_feature_arr是将feature_arr聚类后，得到的三维数组

    for one_clusters_index_arr in clusters_index_arr:
        #以下的连续四行实现了将numpy数组中每个元素变成了int类型（非numpy的int类型）取出来。
        #print(one_clusters_index_arr)
        #print(one_clusters_index_arr[0])
        #for one_index in one_clusters_index_arr[0]:
            #print(type(int(one_index)))
        one_clusters_feature_arr = []
        for one_index in one_clusters_index_arr[0]:
            one_clusters_feature_arr.append(feature_arr[int(one_index)])
        bag_clusters_feature_arr.append(one_clusters_feature_arr)
    # for iiiii in bag_clusters_feature_arr:
    #     print(len(iiiii))
    #print(bag_clusters_feature_arr)
    all_instanceNum = len(feature_arr)
    #print(all_instanceNum)
    bagMatrix = []
    for one_cluster_feature_arr_in_bag in bag_clusters_feature_arr:
        this_one_cluster_instanceNum = len(one_cluster_feature_arr_in_bag)
        #print(this_one_cluster_instanceNum)
        sum_arr = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        #print(len(one_cluster_feature_arr_in_bag))
        for one_feature_arr_in_cluster in one_cluster_feature_arr_in_bag:
            sum_arr = sum_arr + np.array(one_feature_arr_in_cluster)
        ave_arr = sum_arr / float(this_one_cluster_instanceNum)    #计算每个簇的质心
        bagMatrix.append(ave_arr.tolist())
    return bagMatrix

def downBagMatrix(bagMatrix):
    numpy_bagMatrixdown = np.array(bagMatrix).reshape(1,84)       #因为14*6=84，所以reshape(1,84)里的两个参数之和必须为84，否则会报错。
    bagMatrixdown = numpy_bagMatrixdown.tolist()[0]
    return bagMatrixdown

#markedFileDir = 'C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/切滤/beforeEvaRevisedThreshold10/union/CutWithFilteringFeatureFile_F1score.csv'
markedFileDir_left = markedFileDir.split("切滤")[0]
markedFileDir_right = markedFileDir.split("切滤")[1]
corresponding_overall_featureFileDir = markedFileDir_left + "不切" + markedFileDir_right.split("CutWithFilteringFeatureFile")[0] + "NoCutFeatureFile" + markedFileDir_right.split("CutWithFilteringFeatureFile")[1]
corresponding_overall_featureFileObject = open(corresponding_overall_featureFileDir)
try:
     corresponding_overall_featureFile = corresponding_overall_featureFileObject.read( )
finally:
     corresponding_overall_featureFileObject.close( )
corresponding_overall_featureFile_rows = corresponding_overall_featureFile.split('\n')
corresponding_overall_featureFile_rows.pop()

markedFileObject = open(markedFileDir)
try:
     markedFile = markedFileObject.read( )
finally:
     markedFileObject.close( )
markedFile_rows = markedFile.split('\n')
markedFile_rows.pop()
new_markedFileDir_arr = markedFileDir.split("Feature")
new_markedFileDir = new_markedFileDir_arr[0] + "BagFeature" + new_markedFileDir_arr[1]
opened_new_markedFileDir = open(new_markedFileDir, 'ab')
i = -1
for one_markedFile_row in markedFile_rows:
    i = i + 1
    corresponding_overall_featureFile_row_arr = corresponding_overall_featureFile_rows[i].split(",")
    opened_new_markedFileDir.write((str(corresponding_overall_featureFile_row_arr[0]) + "," + str(corresponding_overall_featureFile_row_arr[1]) + "," + str(corresponding_overall_featureFile_row_arr[2]) + "," + str(corresponding_overall_featureFile_row_arr[3]) + "," + str(corresponding_overall_featureFile_row_arr[5]) + "," + str(corresponding_overall_featureFile_row_arr[6]) + "," + str(corresponding_overall_featureFile_row_arr[7]) + ",").encode())
    #print(one_markedFile_row)
    #oneBag_allInstance_feature_arr, clusters_index_arr = bagFeatures(one_markedFile_row)
    bagMatrix = bagFeatures(one_markedFile_row)
    if len(bagMatrix) < 6:
        new_bagMatrix = []
        bagMatrix = bagMatrix + bagMatrix + bagMatrix + bagMatrix + bagMatrix + bagMatrix
        for ooo in bagMatrix:
            new_bagMatrix.append(ooo)
            if len(new_bagMatrix) == 6:
                break
        bagMatrix = new_bagMatrix
    bagMatrixdown = downBagMatrix(bagMatrix)
    print(bagMatrixdown)
    for one_bagMatrixdown_element in bagMatrixdown:
        opened_new_markedFileDir.write((str(one_bagMatrixdown_element) + ",").encode())
    opened_new_markedFileDir.write((str(one_markedFile_row.split(',')[-1]) + "\n").encode())
opened_new_markedFileDir.close()