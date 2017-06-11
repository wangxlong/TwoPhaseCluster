#coding:utf-8
'''
main function

'''
from Util import Util
from FirstPhase import FirstPhaseFunction
from SecondPhase import SecondPhaseFunction
from Evaluate import Evaluationfunction
import numpy as np
from Visualization import  SequencePlot
import matplotlib.pyplot as plt
from DistanceCalculate import DistanceFunction
from sklearn.preprocessing import StandardScaler

globalstatenumber=3

class SequenceExpeiment():
    '序列方案实验,接收参数X，表示所有个体的数据垒在一起的序列'
    
    def __init__(self):
        self.firstPhaseFunction=FirstPhaseFunction()
    
    def onecase(self,X):
        '方案一，fuzzy cmeans 生成序列'
        clusterlabel=self.firstPhaseFunction.fcmeansSe(globalstatenumber,X)
        return clusterlabel
    
    def twocase(self,X):
        '方案二，kmeans 生成序列'
        clusterlabel=self.firstPhaseFunction.kmeansSe(globalstatenumber,X)
        return clusterlabel
    
    def threecase(self,X):
        '方案三，GMM 生成序列'
        clusterlabel=self.firstPhaseFunction.GMMSe(globalstatenumber,X)
        return clusterlabel
    
    def fourcase(self,X):
        '方案四，HMM 生成序列'
        clusterlabel=self.firstPhaseFunction.HMMSe(globalstatenumber,X)
        return clusterlabel
    
    def fivecase(self,X):
        '方案五，gmm－Hmm 生成序列'
        clusterlabel=self.firstPhaseFunction.gmmHmmSe(globalstatenumber,X)
        return clusterlabel

class ClusterExpeiment():
    '''聚类结果实验
    1、path-表示数据的位置，piecewise－分片向量，percence－权重百分百
    2、接收参数X，表示所有个体的数据垒在一起的序列'
    '''
    def __init__(self):
        self.firstPhaseFunction=FirstPhaseFunction()
        self.secondPhaseFunction=SecondPhaseFunction()
        self.distanceFunction=DistanceFunction()
        
    def onecase(self,path,piecewise,features):
        '方案一，paa＋欧式距离+层次'
        data,individul=self.firstPhaseFunction.paadeal(path, piecewise,features)#paa 处理，返回的是dataframe list
        dismat=self.distanceFunction.euclidean(data) # 得到距离矩阵  
        self.secondPhaseFunction.hieracy(dismat, individul, 'average','paa')#层次聚类
    
    def twocase(self,path,piecewise,features,component):
        '方案二，paa＋pca+欧式距离＋层次'
        data,individul=self.firstPhaseFunction.paadeal(path, piecewise,features)#paa 处理，返回的是dataframe list
#         print "data",data[1]
#         print len(data)
        pcadatalist,weightlist=self.firstPhaseFunction.pcadeal(data,component)
        dismat=self.distanceFunction.pcaEuclidean(pcadatalist, weightlist, component)# 得到距离矩阵 
        self.secondPhaseFunction.hieracy(dismat, individul, 'average','paa-pca-ed')#层次聚类

        
    def two1case(self,path,piecewise,features,component):
        '方案二1，paa＋pca+余弦距离＋层次'
        data,individul=self.firstPhaseFunction.paadeal(path, piecewise,features)#paa 处理，返回的是dataframe list
        pcadatalist,weightlist=self.firstPhaseFunction.pcadeal(data,component)
        dismat=self.distanceFunction.pcaCosion(pcadatalist, weightlist, component)# 得到距离矩阵
        self.secondPhaseFunction.hieracy(dismat, individul, 'average','paa-pca-cos')#层次聚类
        
    
    def threecase(self,X):
        '方案三，fcmeans＋层次' 
        clusterlabel=self.firstPhaseFunction.fcmeansSe(globalstatenumber,X)#第一阶段
        stateserieslist,indivdualLabel=Util.divideSeries(clusterlabel)#得到状态序列列表
        distancemat=self.distanceFunction.kl(globalstatenumber, stateserieslist)#得到状态序列列表所对应的距离矩阵
        self.secondPhaseFunction.hieracy(distancemat, indivdualLabel, 'average', 'fcmeans')#层次聚类
        
    def fourcase(self,X):
        '方案四，kmeans＋层次' 
        clusterlabel=self.firstPhaseFunction.kmeansSe(globalstatenumber,X)#第一阶段
        stateserieslist,indivdualLabel=Util.divideSeries(clusterlabel)#得到状态序列列表
        distancemat=self.distanceFunction.kl(globalstatenumber, stateserieslist)#得到状态序列列表所对应的距离矩阵
        self.secondPhaseFunction.hieracy(distancemat, indivdualLabel, 'average', 'kmeans')#层次聚类
    
    def fivecase(self,X):
        '方案五，gmm＋层次'
        clusterlabel=self.firstPhaseFunction.GMMSe(globalstatenumber,X)#第一阶段
        stateserieslist,indivdualLabel=Util.divideSeries(clusterlabel)#得到状态序列列表
        distancemat=self.distanceFunction.kl(globalstatenumber, stateserieslist)#得到状态序列列表所对应的距离矩阵
#         print 'gmm distance:',distancemat
        self.secondPhaseFunction.hieracy(distancemat, indivdualLabel, 'average', 'gmm')#层次聚类
        
    def sixcase(self,X):
        '方案六，hmm＋层次'
        clusterlabel=self.firstPhaseFunction.HMMSe(globalstatenumber,X)#第一阶段
        stateserieslist,indivdualLabel=Util.divideSeries(clusterlabel)#得到状态序列列表
#         print 'hmm stateserieslist:',stateserieslist
        distancemat=self.distanceFunction.kl(globalstatenumber, stateserieslist)#得到状态序列列表所对应的距离矩阵
#         print 'hmm distance:',distancemat
        self.secondPhaseFunction.hieracy(distancemat, indivdualLabel, 'average', 'hmm')#层次聚类
      
    def sevencase(self,X):
        '方案七，gmmhmm＋层次'
        clusterlabel=self.firstPhaseFunction.gmmHmmSe(globalstatenumber,X)#第一阶段
        stateserieslist,indivdualLabel=Util.divideSeries(clusterlabel)#得到状态序列列表
        distancemat=self.distanceFunction.kl(globalstatenumber, stateserieslist)#得到状态序列列表所对应的距离矩阵
#         print 'gmmhmm distance:',distancemat
        self.secondPhaseFunction.hieracy(distancemat, indivdualLabel, 'average', 'gmmhmm')#层次聚类
    
if __name__=='__main__':
    print 'run'
    
    #数据准备阶段
#     features=['Rf','VT','VE','VO2','VCO2','O2exp','CO2exp','VO2/Kg','R','HR','FetO2','FetCO2','Ti','Te','Ttot','Ti/Ttot',
#               'IV','PetO2','PetCO2','PaCO2','PAO2','VD(phys)','VD/VT','EEm','EEbsa','EEkg','npRQ','t Rel',
#               'METS','Qt','SV','predVO2','BR','EEtot']#特征列表
    features=['Rf','VCO2','VO2/Kg','R','EEkg','npRQ','METS','BR']#特征列表
    dataPath='../cleaneddata/'
    allSeries=Util.allIndivdualToOneSeries(dataPath, features)#将所有的序列合为一起
    allSeries=np.array(allSeries)   
    #归一化
    standardScaler=StandardScaler()
    standardallSeries=standardScaler.fit_transform(allSeries)
    
    #实验一，序列spi—value
    evaluationfunction=Evaluationfunction()
    sequenceExpeiment=SequenceExpeiment()
    sequencePlot=SequencePlot()
    
#     label=sequenceExpeiment.onecase(standardallSeries)
#     sequencePlot.plotHidden_State(label, 'fuzzy cmeans')
#     print 'fuzzy cmeans spi_value:',evaluationfunction.spivalue(allSeries, label, globalstatenumber)
    
    label=sequenceExpeiment.twocase(standardallSeries)
    sequencePlot.plotHidden_State(label, 'k-means')
    print 'kmeans spi_value:',evaluationfunction.spivalue(allSeries, label, globalstatenumber)
    
    label=sequenceExpeiment.threecase(standardallSeries)
    sequencePlot.plotHidden_State(label, 'gmm')
    print 'GMM spi_value:',evaluationfunction.spivalue(allSeries, label, globalstatenumber)
    
    label=sequenceExpeiment.fourcase(standardallSeries)
    sequencePlot.plotHidden_State(label, 'hmm')
    print 'HMM spi_value:',evaluationfunction.spivalue(allSeries, label, globalstatenumber)
    
    label=sequenceExpeiment.fivecase(standardallSeries)
    sequencePlot.plotHidden_State(label, 'gmmhmm')
    print 'GMM-hmm spi_value:',evaluationfunction.spivalue(allSeries, label, globalstatenumber)
    
    #实验二，聚类结果
    paafeatures=['Rf','VCO2','VO2/Kg','R','EEkg','npRQ','METS','BR','Speed']#特征列表
    clusterExpeiment=ClusterExpeiment()
    
    clusterExpeiment.onecase('../cleaneddata/',[1,1,1,1,1,1,1,1,1],paafeatures)
    clusterExpeiment.twocase('../cleaneddata/',[1,1,1,1,1,1,1,1,1],paafeatures,3)
    
    clusterExpeiment.two1case('../cleaneddata/',[1,1,1,1,1,1,1,1,1],paafeatures,3)
#     clusterExpeiment.threecase(standardallSeries)
    clusterExpeiment.fourcase(standardallSeries)
    clusterExpeiment.fivecase(standardallSeries)
    clusterExpeiment.sixcase(standardallSeries)
    clusterExpeiment.sevencase(standardallSeries)
  
    plt.show()
    
    
    
    