#coding:utf-8
'''
Created on 2017年2月15日

@author: whl
'''
import os
import csv
import time
import copy
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import  preprocessing
from sklearn.manifold import TSNE,MDS
from sklearn.metrics import euclidean_distances
class Util:
    '工具类'
    
    #读取一个文件夹下面的所有文件
    @staticmethod
    def readAllFileNameFromDir(dataPath):
        filenameList=os.listdir(dataPath)
        #     print (filenameList)
        return filenameList
    
    '从excel中读取数据'
    @staticmethod
    def readDataFromExcel(datapath):
        return pd.read_excel(datapath)       
    
    @staticmethod
    def allIndivdualToOneSeries(dataPath,featurelist):
        labelDic={}
        'dataPath-数据路径，featurelist－特征列表'
        '将所有的序列前后累叠变成一条序列'
        fileList=Util.readAllFileNameFromDir(dataPath)#读取原始数据的文件列表
#     print len(fileList)
        firstPhaseInputData=pd.DataFrame(columns=featurelist)
#     print 'firstPhaseInputData:',firstPhaseInputData
        #循环读取文件
        for i in range(len(fileList)):
            data=pd.read_excel(dataPath+fileList[i])
            data=data.loc[:,featurelist]#取所需要的特征
            print len(data)
#             print fileList[i].decode('utf-8')
            labelDic[fileList[i]]=len(data)
            firstPhaseInputData=firstPhaseInputData.append(data,ignore_index=True)
#     print firstPhaseInputData
        print len(firstPhaseInputData)
        labelPd=pd.Series(labelDic)
        
        labelPd.to_csv("../mediumdata/userLabel.csv") #label读出
        firstPhaseInputData.dropna()#删除空值
        firstPhaseInputData.to_excel("../mediumdata/firstPhaseInputData.xls") #写出最后的数据
        return firstPhaseInputData
    
    @staticmethod
    def divideSeries(clusterLabel):
        '将所有的序列分开'
        'clusterlabel-整条序列，返回分开后的序列列表'
        userflag=pd.read_csv('../mediumdata/userLabel.csv',header=None,names=['name','index'])
        maxLengthIs=np.max(userflag['index'])
#         print maxLengthIs
        beforeindex=lastindex=0
        clusterLabeldic={}
        timeSeriesList=[]
        indivdualLabel=[]
        for i in range(len(userflag)):
            indivdualLabel.append(userflag['name'][i][:-4])        
            lastindex=lastindex+userflag['index'][i]
            everyclusterLabel=clusterLabel[beforeindex:lastindex]
            beforeindex=beforeindex+userflag['index'][i]
#             print type(everyclusterLabel)
            everyclusterLabel=everyclusterLabel.tolist()
            timeSeriesList.append(everyclusterLabel) #把所有的序列存在timeSeriesList中
            
#           #为了写出excel，通过数据补齐，然后构造dataframe结构，然后写出excel            
            outeveryclusterLabel = copy.deepcopy(everyclusterLabel)  #深拷贝，直接复值是浅拷贝
            for j in range(maxLengthIs-len(outeveryclusterLabel)):#等长补齐
                outeveryclusterLabel.append(None)
            clusterLabeldic[userflag['name'][i]]=outeveryclusterLabel
#             print len(everyclusterLabel)
        clusterlabeldf=pd.DataFrame(clusterLabeldic)
        clusterlabeldf.to_csv('../mediumdata/indivdualclusterLabel.csv')
#         print timeSeriesList
        #返回分开后的序列列表
        return timeSeriesList,indivdualLabel
        
    '0-1均值化函数'
    @staticmethod
    def standscalarData(data):
        model=preprocessing.StandardScaler()
        x_scalardata=model.fit_transform(data)
        return x_scalardata
    
    '将datalist写出成csv文件，filename为文件名'
    @staticmethod
    def createListCSV(fileName, dataList):
        with open(fileName, "wb") as csvFile:
            csvWriter = csv.writer(csvFile)
            for data in dataList:
                csvWriter.writerow(data)
            csvFile.close
    @staticmethod
    def createListCSVNoCover(fileName, dataList):
        with open(fileName, "a+") as csvFile:
            csvWriter = csv.writer(csvFile)
            for data in dataList:
                csvWriter.writelines(data)
            csvFile.close
    't-sne可视化'
    @staticmethod
    def tsne_Visition(self,data):
#         data=Util.standscalarData(data)
        print data
        model = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        data=model.fit_transform(data)
        plt.scatter(data[:, 0], data[:, 1],c='r',label='t-sne')
        return plt
    
    'mds-可视化，data为原始多维数据，filename为要将mds转换后的输出写出的路径'
    @staticmethod
    def mds_visition(data,filename,dim):    
        begintime=time.time()    
#         similarities=euclidean_distances(data)
#         print similarities
#         seed = np.random.RandomState(seed=3)
#         mds = MDS(n_components=dim, max_iter=100, eps=1e-9, random_state=seed,
#                    dissimilarity="precomputed", n_jobs=1)
#         x_mds = mds.fit(similarities).embedding_
        mds = MDS(dim, max_iter=600, n_init=1)
        x_mds = mds.fit_transform(data) 
        Util.createListCSV('mds_visision/'+filename,x_mds) # 写出
        endtime=time.time()
        
        print 'mds处理耗时：',endtime-begintime
#         print x_mds
#         plt.scatter(x_mds[:, 0], x_mds[:, 1], color='b', lw=0, label='MDS')
        return x_mds #返回
        
        