# coding=utf-8
import arff          # liac-arff 2.x，返回 dict，非 scipy.io.arff
import xlsxwriter
# import xlrd
import openpyxl
from openpyxl import load_workbook, Workbook
import random
# import sys
# import xlwt
import numpy as np
import os
from streamlit import table

#############################################
np.seterr(divide='ignore', invalid='ignore')
import warnings

warnings.filterwarnings('ignore')
#############################################
from xlutils.copy import copy
from skmultilearn.adapt import MLkNN
from sklearn.model_selection import cross_val_score
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from scipy.sparse import csc_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
import skfeature.utility.entropy_estimators as ees
from numpy import linalg as LA
import skfeature.function.sparse_learning_based.RFS as rfs
import skfeature.function.information_theoretical_based.MRMR as MRMR
import skfeature.function.similarity_based.trace_ratio as trace_ratio
import skfeature.function.similarity_based.fisher_score as fisher_score
import time
import pandas as pd
import scipy
import scipy.io
import csv

from best_parameters import LLSF_best
from LLSF import LLSF
from new import LSMFS_BiSearch

import new


def read_discrete_arff(name,n_features):
    data = arff.load(open(name, 'r'))
    D=np.array(data['data'])
    D=D.astype(dtype='int')
    X=D[:,:n_features]
    mm,nn=np.shape(D)
    yn=nn-n_features
    print("yn=",yn)
    y=np.zeros((mm,yn))
    for i in range(yn):
        y[:,i]=D[:,n_features+i]
    return X,y


def read_numeric_arff(name,n_features):
    data = arff.load(open(name, 'r'))
    D=np.array(data['data']) 
    X=D[:,:n_features]
    Y=D[:,n_features:]
    mm,nn=np.shape(X)
    ym,yn=np.shape(Y)
    FX=np.zeros((mm,nn))
    YY=np.zeros((ym,yn))
    for i in range(mm):
        FX[i, :] = list(map(float, X[i, :]))
        YY[i, :] = list(map(int, Y[i, :]))
    XX=discrete_3(FX)
    print("X的大小=", np.shape(XX))
    print("Y的大小=", np.shape(Y))
    return XX, YY


def read_mat(filename,n_feature):

    mat = scipy.io.loadmat(filename)
    X=mat['bags']
    y = mat['targets']
    print(type(X))
    xx=X[0,0]
    samples,mm=np.shape(X)
    nn,n_feature=np.shape(xx)
    print("samples=",samples)
    print("n_feature=",n_feature)
    XX=np.zeros((samples,n_feature))
    for i in range(samples):
        XX[i]=X[i,0]
    yy=y.T
    rows,los=np.shape(yy)
    for i in range(rows):
        for j in range(los):
            if yy[i,j]==-1:
                yy[i,j]=0
            else:
                yy[i,j]=1
    XX = discrete_3(XX)  
    return XX,yy


def readexcel(filename):

    workbook = openpyxl.load_workbook(filename, data_only=True)
    worksheet = workbook['Sheet1']
    arr = []

    for row in worksheet.iter_rows(values_only=True):
        arr.append(row)
    arr = np.array(arr)
    return arr

def read_xlsx(filename,n_features):

    arr=readexcel(filename)
    X=arr[:,:n_features]
    y=arr[:,n_features:]
    print("数据集大小=",np.shape(X))
    return X,y

def read_data(X,F):

    data=np.zeros((np.shape(X)[0],len(F)))
    z=0
    for i in F:
        i=int(float(i))
        data[:,z]=X[:,i]
        z=z+1
    return data

def readcsv(name,n_features):
    a = open(name, "r")
    lines=csv.reader(a)
    X=[]
    y=[]
    k=0
    for line in lines:
        if k>0:
            sample=list(map(int,line[:n_features]))
            label=list(map(int,line[n_features:]))
            X.append(sample)
            y.append(label)
        k=1
    XX=np.array(X)
    yy=np.array(y)
    print(np.shape(XX))
    print(np.shape(yy))
    return XX,yy


def readF(filename, algnub, index):

    workbook = openpyxl.load_workbook(filename)
    table = workbook['Sheet1']  
    arr = []


    print(table.cell(row=index, column=1).value)

    for i in range(algnub):
        row_values = [table.cell(row=i + index, column=j + 2).value for j in range(table.max_column - 1)]
        arr.append(row_values)

    arr = np.array(arr)
    return arr


def writedata(filname,data):

    m,n=np.shape(data)
    workbook=xlsxwriter.Workbook(filname)
    worksheet=workbook.add_worksheet()
    for i in range(m):
        for j in range(n):
            worksheet.write(data[i,j])
    workbook.close()

def discrete_3(X):

    sample_row, sample_colum = np.shape(X)
    for i in range(sample_colum):
        m = pd.cut(X[:, i], 3, labels=[0, 1, 2])
        for j in range(sample_row):
            X[j, i] = m[j]
    return X


from openpyxl import load_workbook, Workbook


def write_excel_max(filename, dataname, Is, start_pos, flag, alg_nub):
    """
    :param filename: File name
    :param dataname: Dataset name
    :param Is: The two-dimensional array of results to be saved
    :param start_pos: The position in the spreadsheet where the current dataset is stored
    :param flag: Indicates whether this is the first time the spreadsheet is being created
    :param alg_nub: Number of algorithms
    :return: Saves the maximum evaluation values for each algorithm to an Excel file
    """
    alg = ["LSMFS_BiSearch"]
    leng = len(alg)

    if flag == 1:  
        workbook = Workbook()  
        worksheet = workbook.active  
        worksheet.title = "Sheet1"  
        worksheet.cell(row=start_pos+1, column=1, value=dataname)  
        for i in range(leng):
            worksheet.cell(row=start_pos, column=2 + i, value=alg[i]) 
        for i in range(2):
            for j in range(1, alg_nub + 1):
                worksheet.cell(row=start_pos + 1 + i, column=j + 1, value=float(Is[i][j - 1]))  
        workbook.save(filename)  
    else:
      
        workbook = load_workbook(filename)
        worksheet = workbook.active  
        worksheet.cell(row=start_pos+1, column=1, value=dataname)  
        for i in range(leng):
            worksheet.cell(row=start_pos, column=2 + i, value=alg[i])  
        for i in range(2):
            for j in range(1, alg_nub + 1):
                worksheet.cell(row=start_pos + 1 + i, column=j + 1, value=float(Is[i][j - 1]))  
        workbook.save(filename)  


def writeperform(filename, dataname, Is, start_pos, flag, select_nub):
    """
     Function: Saves the algorithm stability results to an Excel spreadsheet;
         Saves the results of accuracy, F1 score, and AUC metrics calculated using classifiers (1NN/3NN/NB, SVC/DT) to an Excel spreadsheet
     Parameters: filename=file name  dataname=dataset name  Is=two-dimensional array of results to be saved
         start_pos=position in the spreadsheet where the current dataset is stored   flag=indicates whether the spreadsheet is being created for the first time
         select_nub=number of features selected for the current dataset (<=30)
    """
    alg = ["LSMFS_BiSearch"]
    leng = len(alg)
    if flag == 1:  
        workbook = Workbook() 
        worksheet = workbook.active
        worksheet.title = "Sheet1"  
        worksheet.cell(row=start_pos, column=1, value=dataname)  
        for i in range(leng):
            worksheet.cell(row=start_pos + i+1, column=1, value=alg[i])  
        for i in range(leng):
            for j in range(1, select_nub+1):
                worksheet.cell(row=start_pos + i+1, column=j + 1, value=float(Is[i][j - 1]))  
        workbook.save(filename)

  
    else:
        workbook = load_workbook(filename)
        worksheet = workbook.active
        worksheet.cell(row=start_pos, column=1, value=dataname) 
        for i in range(leng):
            worksheet.cell(row=start_pos + i+1, column=1, value=alg[i])  
        for i in range(leng):
            for j in range(1, select_nub + 1):
                worksheet.cell(row=start_pos + i+1, column=j + 1, value=float(Is[i][j - 1])) 
        workbook.save(filename)  


def writeperform_obj(filename, dataname, Is, start_pos, flag, select_nub):
    """
     Function: Saves the algorithm stability results to an Excel spreadsheet;
         Stores the results of accuracy, F1 score, and AUC metrics calculated using classifiers (1NN/3NN/NB, SVC/DT) in an Excel spreadsheet
     Parameters: filename=file name  dataname=dataset name  Is=two-dimensional array of results to be saved
         start_pos=position in the spreadsheet where the current dataset is stored   flag=indicates whether the spreadsheet is being created for the first time
         select_nub=number of features selected for the current dataset (<=30)
    """
    alg = ["LSMFS_BiSearch"]
    leng = len(alg)
    if flag == 1: 
        workbook = Workbook()  
        worksheet = workbook.active  
        worksheet.title = "Sheet1"  
        worksheet.cell(row=1, column=start_pos, value=dataname) 
        for i in range(leng):
            worksheet.cell(row=1, column=start_pos + i + 1, value=alg[i])  
        for i in range(leng):
            for j in range(1, select_nub + 1):
                worksheet.cell(row=j + 1, column=start_pos + i + 1, value=float(Is[i][j - 1]))  
        workbook.save(filename)  
    else:
        # 读取现有的 Excel 文件
        workbook = load_workbook(filename)
        worksheet = workbook.active  
        worksheet.cell(row=1, column=start_pos, value=dataname) 
        for i in range(leng):
            worksheet.cell(row=1, column=start_pos + i + 1, value=alg[i]) 
        for i in range(leng):
            for j in range(1, select_nub + 1):
                worksheet.cell(row=j + 1, column=start_pos + i + 1, value=float(Is[i][j - 1])) 
        workbook.save(filename)  

def data_centre(data):
    X = data.T  
    D, N = np.shape(X)  
    mean_X = np.mean(X, 1)
    res = X - np.tile(mean_X, (N, 1)).T  
    R = res.T
    return R

def grid_search(X_train,y_train):
    best_score = 0.0
    for C in [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]:
        clf = BinaryRelevance(svm.LinearSVC(C=C))
        scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1, scoring='accuracy')
        score = scores.mean()  
        if score > best_score:
            best_score = score
            best_parameters = {"C":C}
    C=best_parameters["C"]
    return C

# def save_para(CCCC,i,filename):
#     """
#     :param CCCC: 对应算法的某一最佳参数
#     :param i: 第i个数据集编号
#     :param filename: 文件路径下的文件名
#     :return: 保存最佳结果之后再次返回最佳结果
#     """
#     svmb = str(CCCC)
#     svmc = ["0"]
#     svmc[0] = svmb
#     """注释以下两行"""
#     with open(filename, "a") as file:
#         file.write(svmc[0] + "\n")

#     with open(filename, "r") as file1:
#         save_C = []
#         for l in file1:
#             svmd = l.rstrip()
#             save_C.append(svmd)
#     CCCC = float(save_C[i])
#     return CCCC

def save_para(value, global_idx, filename):
    """
    Append running time / parameter to file.
    The position in file no longer assumes full dataset order.
    """
    with open(filename, "a") as f:
        f.write(f"{global_idx},{value}\n")

    return value

DATASETS = [
    {"name": "flags","train": "flags-train.arff","test": "flags-test.arff","n_features": 19,"type": "numeric"},
    {"name": "birds","train": "birds-train.arff","test": "birds-test.arff","n_features": 260,"type": "numeric"},
    {"name": "emotions","train": "emotions-train.arff","test": "emotions-test.arff","n_features": 72,"type": "numeric"},
    {"name": "genbase","train": "genbase-train.arff.csv","test": "genbase-test.arff.csv","n_features": 1185,"type": "csv"},
    {"name": "medical","train": "medical-train.arff","test": "medical-test.arff","n_features": 1449,"type": "discrete"},
    {"name": "scene","train": "scene-train.arff","test": "scene-test.arff","n_features": 294,"type": "numeric"},
    {"name": "yeast","train": "yeast-train.arff","test": "yeast-test.arff","n_features": 103,"type": "numeric"},
    {"name": "enron","train": "enron-train.arff","test": "enron-test.arff","n_features": 1001,"type": "discrete"},
    {"name": "Arts","train": "Arts_data.mat","test": "Arts_test.mat","n_features": 462,"type": "mat"},
    {"name": "Business","train": "Business_data.mat","test": "Business_test.mat","n_features": 438,"type": "mat"},
    {"name": "Computers","train": "Computers_data.mat","test": "Computers_test.mat","n_features": 681,"type": "mat"},
    {"name": "Education","train": "Education_data.mat","test": "Education_test.mat","n_features": 550,"type": "mat"},
    {"name": "Entertain","train": "Entertain_data.mat","test": "Entertain_test.mat","n_features": 640,"type": "mat"},
    {"name": "Health","train": "Health_data.mat","test": "Health_test.mat","n_features": 612,"type": "mat"},
    {"name": "Recreation","train": "Recreation_data.mat","test": "Recreation_test.mat","n_features": 606,"type": "mat"},
    {"name": "Reference","train": "Reference_data.mat","test": "Reference_test.mat","n_features": 793,"type": "mat"},
    {"name": "Science","train": "Science_data.mat","test": "Science_test.mat","n_features": 743,"type": "mat"},
    {"name": "Society","train": "Society_data.mat","test": "Society_test.mat","n_features": 636,"type": "mat"},
    {"name": "Social", "train": "Social_data.mat", "test": "Social_test.mat", "n_features": 1047, "type": "mat"},
]


def cv(alg_nub, run_datasets=None, run_indices=None):

    flag = 1 
    file_train = ["flags-train.arff", "birds-train.arff", "emotions-train.arff", "genbase-train.arff.csv",
                  "medical-train.arff", "scene-train.arff",
                  "yeast-train.arff", "enron-train.arff",
                  "Arts_data.mat", "Business_data.mat", "Computers_data.mat", "Education_data.mat",
                  "Entertainment_data.mat", "Health_data.mat",
                  "Recreation_data.mat", "Reference_data.mat", "Science_data.mat", "Society_data.mat",
                  "Social_data.mat"]

    file_test = ["flags-test.arff", "birds-test.arff", "emotions-test.arff", "genbase-test.arff.csv",
                 "medical-test.arff", "scene-test.arff",
                 "yeast-test.arff", "enron-test.arff",
                 "Arts_test.mat", "Business_test.mat", "Computers_test.mat", "Education_test.mat",
                 "Entertainment_test.mat", "Health_test.mat",
                 "Recreation_test.mat", "Reference_test.mat", "Science_test.mat", "Society_test.mat", "Social_test.mat"]
    file_featureNub = [19, 260, 72, 1185, 1449, 294, 103, 1001, 462, 438, 681, 550, 640, 612, 606, 793, 743, 636, 1047]

    exceldata = ["flags", "birds", "emotions", "genbase", "medical", "scene", "yeast", "enron", "Arts", "Business",
                 "Computers",
                 "Education",
                 "Entertain", "Health", "Recreation", "Reference", "Science", "Society", "Social"]
    datapos = 1
    dataposmax = 1

    if run_datasets is not None:
        dataset_list = [d for d in DATASETS if d["name"] in run_datasets]
    elif run_indices is not None:
        dataset_list = [DATASETS[i] for i in run_indices]
    else:
        dataset_list = DATASETS

    for run_idx, ds in enumerate(dataset_list):
        i = DATASETS.index(ds)

        print(file_train[i])
        if i <= 2 or i == 5 or i == 6:
            X_train, y_train = read_numeric_arff(file_train[i], file_featureNub[i])
            X_test, y_test = read_numeric_arff(file_test[i], file_featureNub[i])
        if i == 3:
            X_train, y_train = readcsv(file_train[i], file_featureNub[i])
            X_test, y_test = readcsv(file_test[i], file_featureNub[i])
        if i == 4 or i == 7:
            X_train, y_train = read_discrete_arff(file_train[i], file_featureNub[i])
            X_test, y_test = read_discrete_arff(file_test[i], file_featureNub[i])
        if i >= 8:
            X_train, y_train = read_mat(file_train[i], file_featureNub[i])
            X_test, y_test = read_mat(file_test[i], file_featureNub[i])
        ##############################################################################################
        aax, aay = X_train.shape
        if aay == 19:
            select_nub = 19
        elif aay == 1449:
            select_nub = int(round(aay * 0.17))
        else:
            select_nub = int(round(aay * 0.2))
        ###############################################################################################
        HLscores = np.zeros((alg_nub, select_nub))
        RLscores = np.zeros((alg_nub, select_nub))
        CVscores = np.zeros((alg_nub, select_nub))
        APscores = np.zeros((alg_nub, select_nub))
        ZLscores = np.zeros((alg_nub, select_nub))
        MIscoresSVM = np.zeros((alg_nub, select_nub))
        MAscoresSVM = np.zeros((alg_nub, select_nub))
        MIscoresGNB = np.zeros((alg_nub, select_nub))
        MAscoresGNB = np.zeros((alg_nub, select_nub))
        MIscores3NN = np.zeros((alg_nub, select_nub))
        MAscores3NN = np.zeros((alg_nub, select_nub))
        MIscores1NN = np.zeros((alg_nub, select_nub))
        MAscores1NN = np.zeros((alg_nub, select_nub))
        max_svm_MI = np.zeros((2, alg_nub))
        max_gnb_MI = np.zeros((2, alg_nub))
        max_3nn_MI = np.zeros((2, alg_nub))
        max_1nn_MI = np.zeros((2, alg_nub))
        max_svm_MA = np.zeros((2, alg_nub))
        max_gnb_MA = np.zeros((2, alg_nub))
        max_3nn_MA = np.zeros((2, alg_nub))
        max_1nn_MA = np.zeros((2, alg_nub))
        #################################################################################
        # y_train = np.matrix(y_train)
        # aaa, bbb = y_train.shape
        # for ii in range(bbb):
        #     y_train = np.array(y_train)
        #     if len(np.unique(y_train[:, ii])) == 1:
        #         mm = random.randint(0, aaa-1)
        #         y_train[mm][ii] = 1  # random.randint(2,10)
        # AY = y_train
        # abx, aby = AY.shape
        # for ai in range(aby):
        #     ak = 0
        #     for aj in AY[:, ai]:
        #         if aj == 1:
        #             ak = ak + 1
        #     if ak > 1:
        #         ac = AY[:, ai].argmax()
        #         acc = AY[:, ai][::-1]
        #         ad = len(acc) - np.argmax(acc) - 1
        #         if (ad / int(round(abx / 5.0)) - ac / int(round(abx / 5.0))) == 0:
        #             am = -1
        #             while (am < 0) or (am > (abx - 1)):
        #                 if (ac >= 0) and (ac <= round(abx / 5.0)):
        #                     am = random.randint(round(ac + abx / 5.0 + 5), abx)
        #                 elif (ac >= (round(abx / 5.0 * 4 - 1))) and (ac <= (abx - 1)):
        #                     am = random.randint(0, round(ac - abx / 5.0 - 5))
        #                 elif (ac > round(abx / 5.0)) and (ac < (round(abx / 5.0 * 4 - 1))):
        #                     lac = ac - round(abx / 5.0)
        #                     rac = ac + round(abx / 5.0)
        #                     lr = random.randint(0, 1)
        #                     if lr == 0:
        #                         am = random.randint(0, lac)
        #                     elif lr == 1:
        #                         am = random.randint(rac, abx - 1)
        #             AY[am][ai] = 1
        #     elif ak == 1:
        #         ac = AY[:, ai].argmax()
        #         am = -1
        #         while (am < 0) or (am > (abx - 1)):
        #             if (ac >= 0) and (ac <= round(abx / 5.0)):
        #                 am = random.randint(round(ac + abx / 5.0 + 5), abx)
        #             elif (ac >= round(abx / 5.0 * 4 - 1)) and (ac <= abx - 1):
        #                 am = random.randint(0, round(ac - abx / 5.0 - 5))
        #             elif (ac > round(abx / 5.0)) and (ac < round(abx / 5.0 * 4 - 1)):
        #                 lac = ac - round(abx / 5.0)
        #                 rac = ac + round(abx / 5.0)
        #                 lr = random.randint(0, 1)
        #                 if lr == 0:
        #                     am = random.randint(0, lac)
        #                 elif lr == 1:
        #                     am = random.randint(rac, abx - 1)
        #         AY[am][ai] = 1
  
        # df = pd.DataFrame(AY)
        # df.to_csv("y_train\\y_train%d.csv" % i, encoding="utf-8")
        AY = pd.read_csv("y_train\\y_train%d.csv" % i, encoding="utf-8", index_col=0)
        y_train = np.array(AY)
        # CCCC = grid_search(X_train, y_train)
        #
        # C_path = "Para_value\\SVM_C\\data.csv"
        # CCCC = save_para(CCCC,i,C_path)

        svm_para = [0.1, 0.1, 0.01, 1, 1, 0.1, 0.01, 0.1,   0.1, 0.01, 0.0001,   0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        CCCC = svm_para[i]
        ###################################################################################
        FF=[]
        # F_value=[]
        start_Ten = time.time()
        AA = LSMFS_BiSearch(X_train, y_train, select_nub) #n_features=select_nub
        end_Ten = time.time()
        running_time = end_Ten - start_Ten
        LSMFS_path = "Para_value\\LSMFS_BiSearch\\running_time.csv"
        save_para(running_time, i, LSMFS_path)
        print("LSMFS_BiSearch selected features =", AA)

        FF.append(AA)

        # F_value = np.array(F_value)                                 
        # F_value = np.matrix(F_value)
        # F_value = np.array(F_value)

        # print(FF   
        FF = np.array(FF)
        zx = np.matrix(FF)
        zx = zx.astype(int)
        FF = np.array(zx)
        train1 = np.array(zx)
        for an in range(alg_nub):
            for k in range(1, select_nub + 1):
                Fk = FF[an, :k]
                data_tr = read_data(X_train, Fk)
                data_te = read_data(X_test, Fk)
                ########################################################################
                # MLkNN
                classifier = MLkNN()
                classifier.fit(data_tr, y_train.astype(int))
                predictions = classifier.predict(data_te)
                y_score = classifier.predict_proba(data_te)
                y_score = y_score.toarray()
                HLscores[an][k - 1] = hamming_loss(y_test, predictions)
                RLscores[an][k - 1] = label_ranking_loss(y_test, y_score)
                CVscores[an][k - 1] = coverage_error(y_test, y_score)
                APscores[an][k - 1] = label_ranking_average_precision_score(y_test, y_score)
                ZLscores[an][k - 1] = zero_one_loss(y_test, predictions)
                ######################################################################
                # classifier=svm.LinearSVC(C=CCCC)
                # qa,qb=y_test.shape
                # W = np.zeros((qa, qb))
                # for j in range(qb):
                #     classifier.fit(data_tr,y_train[:,j])
                #     predictions=classifier.predict(data_te)
                #     A=np.matrix(predictions).T
                #     W[:,j] = A.ravel()
                #     predictions=W
                # MIscores[an][k-1]=f1_score(y_test,predictions, average='micro')
                # MAscores[an][k-1]=f1_score(y_test,predictions, average='macro')
                #######################################################################
                # SVM
                classifier = BinaryRelevance(svm.LinearSVC(C=CCCC))
                classifier.fit(data_tr, y_train)
                predictions = classifier.predict(data_te)
                predictions = predictions.toarray()
                MIscoresSVM[an][k - 1] = f1_score(y_test, predictions, average='micro')
                MAscoresSVM[an][k - 1] = f1_score(y_test, predictions, average='macro')
                # GaussianNB
                classifier = BinaryRelevance(GaussianNB())
                classifier.fit(data_tr, y_train)
                predictions = classifier.predict(data_te)
                predictions = predictions.toarray()
                MIscoresGNB[an][k - 1] = f1_score(y_test, predictions, average='micro')
                MAscoresGNB[an][k - 1] = f1_score(y_test, predictions, average='macro')
                # 3NN
                classifier = KNeighborsClassifier(n_neighbors=3)
                classifier.fit(data_tr, y_train)
                predictions = classifier.predict(data_te)
                MIscores3NN[an][k - 1] = f1_score(y_test, predictions, average='micro')
                MAscores3NN[an][k - 1] = f1_score(y_test, predictions, average='macro')
                # 1NN
                classifier = KNeighborsClassifier(n_neighbors=1)
                classifier.fit(data_tr, y_train)
                predictions = classifier.predict(data_te)
                MIscores1NN[an][k - 1] = f1_score(y_test, predictions, average='micro')
                MAscores1NN[an][k - 1] = f1_score(y_test, predictions, average='macro')
                #####################################################################################
        max_svm_MI[0] = MIscoresSVM.max(1)
        max_svm_MI[1] = MIscoresSVM.argmax(1) + 1
        max_gnb_MI[0] = MIscoresGNB.max(1)
        max_gnb_MI[1] = MIscoresGNB.argmax(1) + 1
        max_3nn_MI[0] = MIscores3NN.max(1)
        max_3nn_MI[1] = MIscores3NN.argmax(1) + 1
        max_1nn_MI[0] = MIscores3NN.max(1)
        max_1nn_MI[1] = MIscores3NN.argmax(1) + 1

        max_svm_MA[0] = MAscoresSVM.max(1)
        max_svm_MA[1] = MAscoresSVM.argmax(1) + 1
        max_gnb_MA[0] = MAscoresGNB.max(1)
        max_gnb_MA[1] = MAscoresGNB.argmax(1) + 1
        max_3nn_MA[0] = MAscores3NN.max(1)
        max_3nn_MA[1] = MAscores3NN.argmax(1) + 1
        max_1nn_MA[0] = MAscores3NN.max(1)
        max_1nn_MA[1] = MAscores3NN.argmax(1) + 1

        print("分类完成")
        writeperform("new\\hamming_loss.xlsx", exceldata[i], HLscores, datapos, flag, select_nub)
        writeperform("new\\label_ranking_loss.xlsx", exceldata[i], RLscores, datapos, flag, select_nub)
        writeperform("new\\coverage_error.xlsx", exceldata[i], CVscores, datapos, flag, select_nub)
        writeperform("new\\average_precision_score.xlsx", exceldata[i], APscores, datapos, flag, select_nub)
        writeperform("new\\zero_one_loss.xlsx", exceldata[i], ZLscores, datapos, flag, select_nub)
        writeperform("new\\micro_average_score_SVM.xlsx", exceldata[i], MIscoresSVM, datapos, flag, select_nub)
        writeperform("new\\macro_average_score_SVM.xlsx", exceldata[i], MAscoresSVM, datapos, flag, select_nub)
        writeperform("new\\micro_average_score_NB.xlsx", exceldata[i], MIscoresGNB, datapos, flag, select_nub)
        writeperform("new\\macro_average_score_NB.xlsx", exceldata[i], MAscoresGNB, datapos, flag, select_nub)
        writeperform("new\\micro_average_score_3NN.xlsx", exceldata[i], MIscores3NN, datapos, flag, select_nub)
        writeperform("new\\macro_average_score_3NN.xlsx", exceldata[i], MAscores3NN, datapos, flag, select_nub)
        writeperform("new\\micro_average_score_1NN.xlsx", exceldata[i], MIscores1NN, datapos, flag, select_nub)
        writeperform("new\\macro_average_score_1NN.xlsx", exceldata[i], MAscores1NN, datapos, flag, select_nub)
        ###################################################################################################
        writeperform("new\\train1.xlsx", exceldata[i], train1, datapos, flag, select_nub)
        # writeperform_obj("new\\obj_function_value.xlsx", exceldata[i], F_value, datapos, flag, iter)
        write_excel_max("new\\MAX_svm_MI.xlsx", exceldata[i], max_svm_MI, dataposmax, flag, alg_nub)
        write_excel_max("new\\MAX_svm_MI.xlsx", exceldata[i], max_svm_MI, dataposmax, flag, alg_nub)
        write_excel_max("new\\MAX_gnb_MI.xlsx", exceldata[i], max_gnb_MI, dataposmax, flag, alg_nub)
        write_excel_max("new\\MAX_3nn_MI.xlsx", exceldata[i], max_3nn_MI, dataposmax, flag, alg_nub)
        write_excel_max("new\\MAX_1nn_MI.xlsx", exceldata[i], max_1nn_MI, dataposmax, flag, alg_nub)
        write_excel_max("new\\MAX_svm_MA.xlsx", exceldata[i], max_svm_MA, dataposmax, flag, alg_nub)
        write_excel_max("new\\MAX_gnb_MA.xlsx", exceldata[i], max_gnb_MA, dataposmax, flag, alg_nub)
        write_excel_max("new\\MAX_3nn_MA.xlsx", exceldata[i], max_3nn_MA, dataposmax, flag, alg_nub)
        write_excel_max("new\\MAX_1nn_MA.xlsx", exceldata[i], max_1nn_MA, dataposmax, flag, alg_nub)
        flag = 0
        datapos = datapos + 1 + alg_nub
        dataposmax = dataposmax + 3  


if __name__ == "__main__":
    start = time.localtime()
    alg_nub = 1  
    cv(alg_nub, run_indices=[0])  
    end = time.localtime()
    print("start time=", time.asctime(start))
    print(" end  time=", time.asctime(end))
