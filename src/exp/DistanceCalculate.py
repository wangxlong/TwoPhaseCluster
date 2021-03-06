#coding:utf-8
'''
距离计算
'''
import math
import sys
from numpy import *
import numpy as np
class SeriesGetDistanceMtr():
 
    def f_kl(self,x,y):
        if ( x and y ):
            return (x*(math.log(x/y))+y*(math.log(y/x)))*0.5
        else: 
            return sys.maxint
        
    #将状态序列转换为状态转换矩阵，statenumber表示状态的个数，stateseriesarray表示所有的状态序列列表
    def stateToProMatrix(self,statenumber,stateserieslist):
        n = len(stateserieslist) #序列的个数
        for i in range(n):
            stateserieslist[i].pop()
        proMat = mat(zeros((statenumber,n))) #新建矩阵
        for j in range(n):
            for k in range(statenumber):
                proMat[k,j] = float(stateserieslist[j].count(k))/float(len(stateserieslist[j]))  
        return proMat,n
    
    #计算两个概率转换矩阵之间的kl距离
    def calSimilarityByProMatrix(self,ProMatrix,n,m):
        'ProMatrix-状态转化矩阵集合，n－个体个数，m－状态个数'
        mat_kl = mat(zeros((n,n))) #新建kl距离矩阵
        for k in range (n-1):
            a0 = np.array(ProMatrix[:,k].flat)
            for j in range (k+1,n):
                a1 = np.array(ProMatrix[:,j].flat)
                for i in range (m):
                    mat_kl[k,j]= mat_kl[k,j]+self.f_kl(a0[i], a1[i])
        #print(mat_kl[k,j])    
        for i in range (n):
            for j in range (i+1,n):
                mat_kl[j,i] = mat_kl[i,j]
 
#         print 'mat_kl=',mat_kl
        return mat_kl
              
    def fit(self,statenumber,stateserieslist):
        'statenumber-状态个数，stateserieslist－状态序列列表,返回－距离矩阵'
        ProMatrix,n=self.stateToProMatrix(statenumber,stateserieslist)
        mat_kl=self.calSimilarityByProMatrix(ProMatrix,n,statenumber)
        return mat_kl

class TwoDataGetDistance():
    '''
    欧式，余炫距离
    '''
    

class DistanceFunction():
    '''
    距离矩阵计算
    '''
    def kl(self,statenumber,stateserieslist):
        'statenumber-状态个数，stateserieslist－状态序列列表,返回－距离矩阵'
        seriesGetDistanceMtr=SeriesGetDistanceMtr()
        mat_kl=seriesGetDistanceMtr.fit(statenumber, stateserieslist)
        return mat_kl
    
    def euclidean(self,data):
        '欧式距离，data是dataframe 列表'
        n=len(data)
        dismat=mat(zeros((n,n)))
        for i in range(n):
            for j in range(n):
                dismat[i,j]=((data[i]-data[j])*(data[i]-data[j])).sum().sum()
            
        return  dismat
    
    def pcaComponetCompute(self,weightlist,percent):
        #计算平均weight
        sumweight=np.zeros((len(weightlist[0])))
        print "weightlist:",weightlist
       
        for i in range(len(weightlist)):
            print "dtype:",len(weightlist[i])
            sumweight=sumweight+weightlist[i]
        sumweight=sumweight/len(weightlist)
        
        count=0
        for i in range(len(sumweight)):
            count=count+sumweight[i]
        print "count:",count
        
        #计算component
        sumpercent=0
        percentValue=percent/100
        for i in range(len(sumweight)):
            sumpercent=sumpercent+sumweight[i]
            if sumpercent>=percentValue:
                return i+1
           
    
    def pcaEuclidean(self,pcadatalist,weightlist,component):
        sumweight=np.zeros((len(weightlist[0])))
#         print "weightlist:",weightlist
        for i in range(len(weightlist)):
#             print "dtype:",len(weightlist[i])
            sumweight=sumweight+weightlist[i]
        sumweight=sumweight/len(weightlist)
        #验证小于1
#         count=0
#         for i in range(len(sumweight)):
#             count=count+sumweight[i]
#         print "count:",count
#         print "pcadatalist:",pcadatalist
#         print component
        n=len(pcadatalist)
        dismat=mat(zeros((n,n)))
        for i in range(n):
            for j in range(n):
#                 print "j is:",pcadatalist[j]
                tmpmatrix=(pcadatalist[i]-pcadatalist[j])*(pcadatalist[i]-pcadatalist[j])
#                 print tmpmatrix
#                 print len(tmpmatrix)
#                 print len(tmpmatrix[0]) 
                tmpsum=0             
                for k in range(len(tmpmatrix)):
                    tmpsum=tmpsum+(tmpmatrix[k]*sumweight).sum()
                
                dismat[i,j]=tmpsum                 
                
        return  dismat
        
        
    def pcaCosion(self,pcadatalist,weightlist,component):
        sumweight=np.zeros((len(weightlist[0])))
#         print "weightlist:",weightlist
        for i in range(len(weightlist)):
#             print "dtype:",len(weightlist[i])
            sumweight=sumweight+weightlist[i]
        sumweight=sumweight/len(weightlist)

        n=len(pcadatalist)
        dismat=mat(zeros((n,n)))
        for i in range(n):
            for j in range(n):
                tmpsum=0             
                for k in range(component):
                    tmpsum=tmpsum+dot(pcadatalist[i][:,k],pcadatalist[j][:,k])/(linalg.norm(pcadatalist[i][:,k])*linalg.norm(pcadatalist[j][:,k]))*sumweight[k]      
                dismat[i,j]=1-abs(tmpsum)                
                
        return  dismat
        
            
            
            
            
            
        
    