# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.distance import directed_hausdorff

#resultFileDir = r"C:\Users\Administrator\Desktop\博士论文\第四个点\数据\新数据\resultFile\\"  #注意前面一定要加上r表示强制不转移，但是最后的斜杠不能只写一个,或者干脆用下面这种写法。
resultFileDir = r"C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/resultFile/"

#此行往上的路径是特征数据路径，此行往下的路径是标签数据路径和生成文件路径

# #最终新法F1score模块开始
# realMarkedFileDir = 'C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/setsLabel/realData/intersection/realDataF1score.csv'
# markedFileDir = 'C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/setsLabel/simData/newEvaluationMethod/new_simRecorededF1score.csv'
# NoCutFeatureFile = open('C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/新标签方法不切/NoCutFeatureFile_F1score.csv', 'ab')
# CutWithFilteringFeatureFile = open('C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/新标签方法切滤/CutWithFilteringFeatureFile_F1score.csv', 'ab')
# #最终新法F1score模块结束

# #最终新法Precision模块开始
# realMarkedFileDir = 'C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/setsLabel/realData/intersection/realDataPrecision.csv'
# markedFileDir = 'C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/setsLabel/simData/newEvaluationMethod/new_simRecorededPrecision.csv'
# NoCutFeatureFile = open('C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/新标签方法不切/NoCutFeatureFile_Precision.csv', 'ab')
# CutWithFilteringFeatureFile = open('C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/新标签方法切滤/CutWithFilteringFeatureFile_Precision.csv', 'ab')
# #最终新法Precision模块结束

#最终新法Recall模块开始
realMarkedFileDir = 'C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/setsLabel/realData/intersection/realDataRecall.csv'
markedFileDir = 'C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/setsLabel/simData/newEvaluationMethod/new_simRecorededRecall.csv'
NoCutFeatureFile = open('C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/新标签方法不切/NoCutFeatureFile_Recall.csv', 'ab')
CutWithFilteringFeatureFile = open('C:/Users/Administrator/Desktop/博士论文/第四个点/数据/新数据/新标签方法切滤/CutWithFilteringFeatureFile_Recall.csv', 'ab')
#最终新法Recall模块结束

realDataName_arr = ["HG001", "HG002", "HG003", "HG004", "HG005_231439", "HG006", "HG007"]
faFeature_arr = ["l_hr_hd", "l_hr_nd", "l_nr_hd", "l_nr_nd", "m_hr_hd", "m_hr_nd", "m_nr_hd", "m_nr_nd", "s_hr_hd", "s_hr_nd", "s_nr_hd", "s_nr_nd"]
readLen_arr = ["1000", "2000", "3000", "5000", "10000", "15000", "20000", "25000"]
readLen_arr_dic = {
"l_hr_hd": ["1000", "2000", "3000", "5000", "10000", "15000", "20000", "25000"],
"l_hr_nd": ["6000", "7000", "8000", "9000", "11000", "12000", "13000", "16000"],
"l_nr_hd": ["1000", "2000", "3000", "5000", "10000", "15000", "20000", "25000"],
"l_nr_nd": ["4000", "9000", "12000", "14000", "17000", "21000", "22000", "24000"],
"m_hr_hd": ["4000", "6000", "8000", "16000", "21000", "22000", "23000", "24000"],
"m_hr_nd": ["6000", "8000", "11000", "13000", "16000", "18000", "21000", "23000"],
"m_nr_hd": ["7000", "8000", "11000", "13000", "17000", "19000", "22000", "23000"],
"m_nr_nd": ["7000", "11000", "13000", "14000", "16000", "17000", "18000", "19000"],
"s_hr_hd": ["9000", "11000", "13000", "16000", "18000", "22000", "23000", "24000"],
"s_hr_nd": ["4000", "6000", "8000", "12000", "18000", "19000", "23000", "24000"],
"s_nr_hd": ["3000", "4000", "6000", "9000", "14000", "18000", "21000", "24000"],
"s_nr_nd": ["4000", "7000", "9000", "12000", "14000", "17000", "19000", "22000"]
}
depth_arr = ["10", "30", "50", "70", "90", "110", "130", "150"]
depth_arr_dic = {
"l_hr_hd": ["10", "30", "50", "70", "90", "110", "130", "150"],
"l_hr_nd": ["1", "3", "5", "7", "9", "11", "13", "15"],
"l_nr_hd": ["1", "3", "5", "7", "9", "11", "13", "15"],
"l_nr_nd": ["1", "3", "5", "7", "9", "11", "13", "15"],
"m_hr_hd": ["1", "3", "5", "7", "9", "11", "13", "15"],
"m_hr_nd": ["10", "30", "50", "70", "90", "110", "130", "150"],
"m_nr_hd": ["1", "3", "5", "7", "9", "11", "13", "15"],
"m_nr_nd": ["1", "3", "5", "7", "9", "11", "13", "15"],
"s_hr_hd": ["1", "3", "5", "7", "9", "11", "13", "15"],
"s_hr_nd": ["10", "30", "50", "70", "90", "110", "130", "150"],
"s_nr_hd": ["1", "3", "5", "7", "9", "11", "13", "15"],
"s_nr_nd": ["10", "30", "50", "70", "90", "110", "130", "150"]
}

def getLabelSet(markedFileDir, realMarkedFileDir):
    label_arr = []

    markedFileObject = open(markedFileDir)
    try:
         markedFile = markedFileObject.read( )
    finally:
         markedFileObject.close( )
    markedFile_rows = markedFile.split('\n')
    markedFile_rows.pop()

    realMarkedFileObject = open(realMarkedFileDir)
    try:
        realMarkedFile = realMarkedFileObject.read()
    finally:
        realMarkedFileObject.close()
    realMarkedFile_rows = realMarkedFile.split('\n')
    realMarkedFile_rows.pop()

    markedFile_rows = markedFile_rows + realMarkedFile_rows
    #print(len(markedFile_rows))
    max_index = 0
    for one_markedFile_row in markedFile_rows:
        one_markedFile_row_arr = one_markedFile_row.split(",")
        one_markedFile_row_arr = [ii for ii in one_markedFile_row_arr if ii != ""]
        iii = 0
        max_value = 0
        for one_markedFile_row_arr_element in one_markedFile_row_arr:
            iii = iii + 1
            if float(one_markedFile_row_arr_element) > max_value:
                max_value = float(one_markedFile_row_arr_element)
                max_index = iii
        label_arr.append(max_index)
    return label_arr

def getSingleLabel(markedFileDir, realMarkedFileDir):
    label_arr = []

    markedFileObject = open(markedFileDir)
    try:
         markedFile = markedFileObject.read( )
    finally:
         markedFileObject.close( )
    markedFile_rows = markedFile.split('\n')
    markedFile_rows.pop()

    realMarkedFileObject = open(realMarkedFileDir)
    try:
        realMarkedFile = realMarkedFileObject.read()
    finally:
        realMarkedFileObject.close()
    realMarkedFile_rows = realMarkedFile.split('\n')
    realMarkedFile_rows.pop()

    markedFile_rows = markedFile_rows + realMarkedFile_rows
    markedFile_rows.pop()
    #print(len(markedFile_rows))
    max_index = 0
    for one_markedFile_row in markedFile_rows:
        one_markedFile_row_arr = one_markedFile_row.split(",")
        one_markedFile_row_arr = [ii for ii in one_markedFile_row_arr if ii != ""]
        iii = 0
        max_value = 0
        #singLable_one_markedFile_row_arr = one_markedFile_row_arr[:6]
        singLable_one_markedFile_row_arr = one_markedFile_row_arr[:5]
        singLable_one_markedFile_row_arr.append(one_markedFile_row_arr[31])
        #print(one_markedFile_row_arr[31])
        for one_markedFile_row_arr_element in singLable_one_markedFile_row_arr:
            iii = iii + 1
            if float(one_markedFile_row_arr_element) > max_value:
                max_value = float(one_markedFile_row_arr_element)
                max_index = iii
        if max_index == 6:
            label_arr.append(32)
        else:
            label_arr.append(max_index)
    return label_arr

def fileToArr(this_file_dir):
    this_fileObject = open(this_file_dir)
    try:
        this_file = this_fileObject.read()
    finally:
        this_fileObject.close()
    this_file_rows = this_file.split('\n')
    this_file_rows.pop()
    return this_file_rows

def intervalDecode(iniPos1, iniPos2):
    str_pos1 = str(iniPos1)
    this_pos1 = int(str_pos1[1:len(str_pos1)])
    this_chr1 = iniPos1 - this_pos1
    str_pos2 = str(iniPos2)
    this_pos2 = int(str_pos2[1:len(str_pos2)])
    this_chr2 = iniPos2 - this_pos2
    decodedInterval = np.array([(this_pos1, this_chr1), (this_pos2, this_chr2)])
    return decodedInterval

def getNoCutFeature(this_file_rows):
    NoCutFeature = this_file_rows[0].split(",")
    return NoCutFeature

def getCutWithFilteringFeature(this_file_rows):
    CutWithFilteringFeature = []
    result_sv_begin_arr = this_file_rows[1].split(",")
    result_sv_end_arr = this_file_rows[2].split(",")
    ifInRepeatRegionArr = this_file_rows[3].split(",")
    allSVs_len_arr = this_file_rows[4].split(",")
    allSVs_readLen_arr = this_file_rows[5].split(",")
    allSVs_depth_arr = this_file_rows[6].split(",")
    max_ins_ratio_arr = this_file_rows[7].split(",")
    max_del_ratio_arr = this_file_rows[8].split(",")
    max_sub_ratio_arr = this_file_rows[9].split(",")
    max_map_ratio_arr = this_file_rows[10].split(",")
    ins_ratio_arr = this_file_rows[11].split(",")
    del_ratio_arr = this_file_rows[12].split(",")
    sub_ratio_arr = this_file_rows[13].split(",")
    map_ratio_arr = this_file_rows[14].split(",")
    for i in range(len(result_sv_begin_arr)):
        if i == 0:
            left_interval = intervalDecode(int(result_sv_begin_arr[i + 1]), int(result_sv_end_arr[i + 1]))
            right_interval = intervalDecode(int(result_sv_begin_arr[i + 1]), int(result_sv_end_arr[i + 1]))
        elif i == len(result_sv_begin_arr) - 1:
            left_interval = intervalDecode(int(result_sv_begin_arr[i - 1]), int(result_sv_end_arr[i - 1]))
            right_interval = intervalDecode(int(result_sv_begin_arr[i - 1]), int(result_sv_end_arr[i - 1]))
        else:
            left_interval = intervalDecode(int(result_sv_begin_arr[i - 1]), int(result_sv_end_arr[i - 1]))
            right_interval = intervalDecode(int(result_sv_begin_arr[i + 1]), int(result_sv_end_arr[i + 1]))
        this_interval = intervalDecode(int(result_sv_begin_arr[i]), int(result_sv_end_arr[i]))
        left_hausdorff = directed_hausdorff(left_interval, this_interval)[0]
        right_hausdorff = directed_hausdorff(this_interval, right_interval)[0]
        CutWithFilteringFeature.append(left_hausdorff)
        CutWithFilteringFeature.append(right_hausdorff)
        CutWithFilteringFeature.append(float(ifInRepeatRegionArr[i]))
        CutWithFilteringFeature.append(float(allSVs_len_arr[i]))
        CutWithFilteringFeature.append(float(allSVs_readLen_arr[i]))
        CutWithFilteringFeature.append(float(allSVs_depth_arr[i]))
        CutWithFilteringFeature.append(float(max_ins_ratio_arr[i]))
        CutWithFilteringFeature.append(float(max_del_ratio_arr[i]))
        CutWithFilteringFeature.append(float(max_sub_ratio_arr[i]))
        CutWithFilteringFeature.append(float(max_map_ratio_arr[i]))
        CutWithFilteringFeature.append(float(ins_ratio_arr[i]))
        CutWithFilteringFeature.append(float(del_ratio_arr[i]))
        CutWithFilteringFeature.append(float(sub_ratio_arr[i]))
        CutWithFilteringFeature.append(float(map_ratio_arr[i]))
    return CutWithFilteringFeature


allFileLabelSet = getLabelSet(markedFileDir, realMarkedFileDir)
allFileSingleLabel = getSingleLabel(markedFileDir, realMarkedFileDir)

allFileLabelSet_index = -1
for one_faFeature in faFeature_arr:
    #print(one_faFeature)
    for one_readLen in readLen_arr:
        for one_depth in depth_arr:
            allFileLabelSet_index = allFileLabelSet_index + 1
            this_file = resultFileDir + "a_" + one_faFeature + "_" + one_readLen + "_" + one_depth + ".csv"
            this_file_rows = fileToArr(this_file)
            this_file_NoCutFeature = getNoCutFeature(this_file_rows)
            for one_this_file_NoCutFeature in this_file_NoCutFeature:
                NoCutFeatureFile.write((str(one_this_file_NoCutFeature) + ",").encode())
            NoCutFeatureFile.write((str(allFileSingleLabel[allFileLabelSet_index]) + "," + str(allFileLabelSet[allFileLabelSet_index]) + "\n").encode())
            this_file_CutWithFilteringFeature = getCutWithFilteringFeature(this_file_rows)
            for one_this_file_CutWithFilteringFeature in this_file_CutWithFilteringFeature:
                CutWithFilteringFeatureFile.write((str(one_this_file_CutWithFilteringFeature) + ",").encode())
            CutWithFilteringFeatureFile.write((str(allFileLabelSet[allFileLabelSet_index]) + "\n").encode())



allFileLabelSet_index = 767
for one_faFeature in faFeature_arr:
    #print(one_faFeature)
    for one_readLen in readLen_arr_dic[one_faFeature]:
        for one_depth in depth_arr_dic[one_faFeature]:
            allFileLabelSet_index = allFileLabelSet_index + 1
            this_file = resultFileDir + "b_" + one_faFeature + "_" + one_readLen + "_" + one_depth + ".csv"
            this_file_rows = fileToArr(this_file)
            this_file_NoCutFeature = getNoCutFeature(this_file_rows)
            for one_this_file_NoCutFeature in this_file_NoCutFeature:
                NoCutFeatureFile.write((str(one_this_file_NoCutFeature) + ",").encode())
            NoCutFeatureFile.write((str(allFileSingleLabel[allFileLabelSet_index]) + "," + str(allFileLabelSet[allFileLabelSet_index]) + "\n").encode())
            this_file_CutWithFilteringFeature = getCutWithFilteringFeature(this_file_rows)
            for one_this_file_CutWithFilteringFeature in this_file_CutWithFilteringFeature:
                CutWithFilteringFeatureFile.write((str(one_this_file_CutWithFilteringFeature) + ",").encode())
            CutWithFilteringFeatureFile.write((str(allFileLabelSet[allFileLabelSet_index]) + "\n").encode())

allFileLabelSet_index = 1535
for one_realDataName in realDataName_arr:
    allFileLabelSet_index = allFileLabelSet_index + 1
    this_file = resultFileDir + one_realDataName + ".csv"
    this_file_rows = fileToArr(this_file)
    this_file_NoCutFeature = getNoCutFeature(this_file_rows)
    for one_this_file_NoCutFeature in this_file_NoCutFeature:
        NoCutFeatureFile.write((str(one_this_file_NoCutFeature) + ",").encode())
    NoCutFeatureFile.write((str(allFileSingleLabel[allFileLabelSet_index]) + "," + str(allFileLabelSet[allFileLabelSet_index]) + "\n").encode())
    this_file_CutWithFilteringFeature = getCutWithFilteringFeature(this_file_rows)
    for one_this_file_CutWithFilteringFeature in this_file_CutWithFilteringFeature:
        CutWithFilteringFeatureFile.write((str(one_this_file_CutWithFilteringFeature) + ",").encode())
    CutWithFilteringFeatureFile.write((str(allFileLabelSet[allFileLabelSet_index]) + "\n").encode())

NoCutFeatureFile.close()
CutWithFilteringFeatureFile.close()