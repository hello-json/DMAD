from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV

from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset

from sklearn import metrics
import numpy as np
from Metrics import RankingLoss, AveragePrecision

from scipy.sparse import csc_matrix

markedFileDir = r"C:\Users\Administrator\Desktop\博士论文\第四个点\数据\新数据\新标签方法切滤\CutWithFilteringBagFeatureFile_Recall.csv"

#下面的realMarkedFileDir和simMarkedFileDir两个变量引入的文件是为了写predicted_setslabel_file_dir文件时查找标签对应的性能值是多少，所以暂且把realMarkedFileDir变量放这儿也不影响只有模拟数据时候的实验，以防万一后面想加入真实数据一起训练。
realMarkedFileDir = 'C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/setsLabel/realData/intersectionAndUnion/realDataRecall.csv'
simMarkedFileDir = 'C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/setsLabel/simData/newEvaluationMethod/new_simRecorededRecall.csv'

predicted_setslabel_file_dir = "C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/新标签方法切滤/predictedSetsLabel_CutWithFilteringBagFeatureFile_Recall.csv"

#X, Y = make_multilabel_classification(n_samples=10, n_features=8, n_classes=5, n_labels=3)   #n_labels代表平均每个样本两个标签
label_dic = {
    1: [1, 0, 0, 0, 0, 0],
    2: [0, 1, 0, 0, 0, 0],
    3: [0, 0, 1, 0, 0, 0],
    4: [0, 0, 0, 1, 0, 0],
    5: [0, 0, 0, 0, 1, 0],
    6: [1, 1, 0, 0, 0, 0],
    7: [1, 0, 1, 0, 0, 0],
    8: [1, 0, 0, 1, 0, 0],
    9: [1, 0, 0, 0, 1, 0],
    10: [0, 1, 1, 0, 0, 0],
    11: [0, 1, 0, 1, 0, 0],
    12: [0, 1, 0, 0, 1, 0],
    13: [0, 0, 1, 1, 0, 0],
    14: [0, 0, 1, 0, 1, 0],
    15: [0, 0, 0, 1, 1, 0],
    16: [1, 1, 1, 0, 0, 0],
    17: [1, 1, 0, 1, 0, 0],
    18: [1, 1, 0, 0, 1, 0],
    19: [1, 0, 1, 1, 0, 0],
    20: [1, 0, 1, 0, 1, 0],
    21: [1, 0, 0, 1, 1, 0],
    22: [0, 1, 1, 1, 0, 0],
    23: [0, 1, 1, 0, 1, 0],
    24: [0, 1, 0, 1, 1, 0],
    25: [0, 0, 1, 1, 1, 0],
    26: [1, 1, 1, 1, 0, 0],
    27: [1, 1, 1, 0, 1, 0],
    28: [1, 1, 0, 1, 1, 0],
    29: [1, 0, 1, 1, 1, 0],
    30: [0, 1, 1, 1, 1, 0],
    31: [1, 1, 1, 1, 1, 0],
    32: [0, 0, 0, 0, 0, 1],
    33: [1, 0, 0, 0, 0, 1],
    34: [0, 1, 0, 0, 0, 1],
    35: [0, 0, 1, 0, 0, 1],
    36: [0, 0, 0, 1, 0, 1],
    37: [0, 0, 0, 0, 1, 1],
    38: [1, 1, 0, 0, 0, 1],
    39: [1, 0, 1, 0, 0, 1],
    40: [1, 0, 0, 1, 0, 1],
    41: [1, 0, 0, 0, 1, 1],
    42: [0, 1, 1, 0, 0, 1],
    43: [0, 1, 0, 1, 0, 1],
    44: [0, 1, 0, 0, 1, 1],
    45: [0, 0, 1, 1, 0, 1],
    46: [0, 0, 1, 0, 1, 1],
    47: [0, 0, 0, 1, 1, 1],
    48: [1, 1, 1, 0, 0, 1],
    49: [1, 1, 0, 1, 0, 1],
    50: [1, 1, 0, 0, 1, 1],
    51: [1, 0, 1, 1, 0, 1],
    52: [1, 0, 1, 0, 1, 1],
    53: [1, 0, 0, 1, 1, 1],
    54: [0, 1, 1, 1, 0, 1],
    55: [0, 1, 1, 0, 1, 1],
    56: [0, 1, 0, 1, 1, 1],
    57: [0, 0, 1, 1, 1, 1],
    58: [1, 1, 1, 1, 0, 1],
    59: [1, 1, 1, 0, 1, 1],
    60: [1, 1, 0, 1, 1, 1],
    61: [1, 0, 1, 1, 1, 1],
    62: [0, 1, 1, 1, 1, 1],
    63: [1, 1, 1, 1, 1, 1],
}
#markedFileDir = r"C:\Users\Administrator\Desktop\博士论文\第三个点\数据\完整数据\只有deletion\新的八个特征\\recorededF1scoreFile.csv"  #注意由于python里\是转义符，所以引号前的r不能少
markedFileObject = open(markedFileDir)
try:
     markedFile = markedFileObject.read( )
finally:
     markedFileObject.close( )
markedFile_rows = markedFile.split('\n')
markedFile_rows.pop()
feature_arr = []
label_arr = []
#First = True      #跳过第一行
for one_markedFile_row in markedFile_rows:
    #if First:                    #跳过第一行
    #    First = False            #跳过第一行
    #    continue                 #跳过第一行
    one_markedFile_row_arr = one_markedFile_row.split(",")
    #feature_arr.append([float(one_markedFile_row_arr[0]), float(one_markedFile_row_arr[1]), float(one_markedFile_row_arr[2]), float(one_markedFile_row_arr[3]), float(one_markedFile_row_arr[4]), float(one_markedFile_row_arr[5]), float(one_markedFile_row_arr[6]), float(one_markedFile_row_arr[7])])
    oneLine_feature_arr = []
    for markedFile_row_arr_element in one_markedFile_row_arr:
        oneLine_feature_arr.append(float(markedFile_row_arr_element))
    oneLine_feature_arr.pop()
    #print(oneLine_feature_arr)
    feature_arr.append(oneLine_feature_arr)
    label_arr.append(label_dic[int(one_markedFile_row_arr[-1])])
X = np.array(feature_arr)       #必须转成numpy数组下面才能正常运行
Y = np.array(label_arr)
#print(X)
#print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9, test_size=0.1)
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1)   #test_size的值如果是小数就是测试集占比，如果是整数就是测试样本个数
#cls = DecisionTreeClassifier()
#cls = ExtraTreeClassifier()
#cls = ExtraTreesClassifier()
#cls = KNeighborsClassifier()
#cls = MLPClassifier()
#cls = RadiusNeighborsClassifier()   #这个跑不起来
#cls = RidgeClassifierCV()   #这个跑不起来
#cls = RandomForestClassifier()

cls = BinaryRelevance(
    classifier = RandomForestClassifier(),
    require_dense = [False, True]
)
# cls = ClassifierChain(
#     classifier = RandomForestClassifier(),
#     require_dense = [False, True]
# )
# cls = LabelPowerset(
#     classifier = RandomForestClassifier(),
#     require_dense = [False, True]
# )

cls.fit(X_train, Y_train)

Y_pred = cls.predict(X_test)
#print(csc_matrix.todense(Y_pred))      #用csc_matrix.todense将csc矩阵进行一个转换。
#print(Y_test)
#print(metrics.f1_score(Y_test, Y_pred, average="macro"))
#print(metrics.f1_score(Y_test, Y_pred, average="micro"))
#print(type(Y_test))
#print(Y_pred)
print(metrics.f1_score(Y_test, Y_pred, average="weighted"))
#print(metrics.f1_score(Y_test, Y_pred, average="samples"))
#print(metrics.roc_auc_score(Y_test, Y_pred))    #只有前面几个cls有AUC值，BR、CC、LP三个框架没有AUC值。
print(metrics.hamming_loss(Y_test, Y_pred))
print(metrics.accuracy_score(Y_test, Y_pred))
#print(AveragePrecision(Y_test, Y_pred))   #这个也不能用
#多标签框架下不能用的两个指标开始
#print(metrics.label_ranking_loss(Y_test, Y_pred))
#print(RankingLoss(Y_test, Y_pred))
#print(metrics.coverage_error(Y_test, Y_pred))
#多标签框架下不能用的两个指标结束
print(metrics.zero_one_loss(Y_test, Y_pred))   #（该网址有这几个指标的公式！！！）从https://blog.csdn.net/mr_muli/article/details/96451565来看，这个应该就是oneError
#Y_prob = cls.predict_proba(X_test)
#print(Y_prob)


# simMarkedFileObject = open(simMarkedFileDir)
# try:
#      simMarkedFile = simMarkedFileObject.read( )
# finally:
#      simMarkedFileObject.close( )
# simMarkedFile_rows = simMarkedFile.split('\n')
# simMarkedFile_rows.pop()
#
# realMarkedFileObject = open(realMarkedFileDir)
# try:
#     realMarkedFile = realMarkedFileObject.read()
# finally:
#     realMarkedFileObject.close()
# realMarkedFile_rows = realMarkedFile.split('\n')
# realMarkedFile_rows.pop()
#
# opened_predicted_setslabel_file = open(predicted_setslabel_file_dir, 'ab')
# for i in range(len(X)):
#     print(i)
#     this_split_X = np.split(X, (i, i + 1))
#     this_split_Y = np.split(Y, (i, i + 1))
#     this_X_train = np.concatenate([this_split_X[0], this_split_X[2]], axis=0)
#     this_X_test = this_split_X[1]
#     this_Y_train = np.concatenate([this_split_Y[0], this_split_Y[2]], axis=0)
#     this_Y_test = this_split_Y[1]
#     cls.fit(this_X_train, this_Y_train)
#     this_Y_pred = csc_matrix.todense(cls.predict(this_X_test)).tolist()[0]
#     this_Y_pred_key = list(label_dic.keys())[list(label_dic.values()).index(this_Y_pred)]   #根据value找key值语句!!!
#     #print(this_Y_pred)
#     #print(this_Y_pred_key)
#     opened_predicted_setslabel_file.write((str(this_Y_pred_key) + ",").encode())
#     if i > 1535:
#         opened_predicted_setslabel_file.write((realMarkedFile_rows[i - 1536].split(',')[int(this_Y_pred_key) - 1] + "\n").encode())
#     else:
#         opened_predicted_setslabel_file.write((simMarkedFile_rows[i].split(',')[int(this_Y_pred_key) - 1] + "\n").encode())
# opened_predicted_setslabel_file.close()
