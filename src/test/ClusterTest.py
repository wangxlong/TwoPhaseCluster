#coding:utf-8

from exp.exp import ClusterExpeiment
import numpy as np
from exp.Util import Util
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

if __name__=='__main__':
    
    #数据准备
    features=['Rf','VCO2','VO2/Kg','R','EEkg','npRQ','METS','Speed']#特征列表
    dataPath='../cleaneddata/'
    #归一化
#     standardScaler=StandardScaler()
#     standardallSeries=standardScaler.fit_transform(allSeries)
    
    # 测试    
    clusterExpeiment=ClusterExpeiment()
    clusterExpeiment.twocase('../cleaneddata/',[1,1,1,1,1,1,1,1,1],features,3)
    
    clusterExpeiment.two1case('../cleaneddata/',[1,1,1,1,1,1,1,1,1],features,3)
    
    plt.show()