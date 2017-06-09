#coding:utf-8
'''
second phase code

'''
from numpy import *
import numpy as np 
import time
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

class KMEDOIDS():
              
    def total_cost(self,dist_mat, medoids):
        '''
        compute the total cost based on current setting.
        '''
        med_idx = medoids[-1];
        k = len(med_idx);
        cost = 0.0;
        dis = dist_mat[:,med_idx[0]]
        dis = dis[:,None] #将一维数组转化为二维数组的一列

        for i in range(1,k):
            d0 = dist_mat[:,med_idx[i]]
            d0 = d0[:,None] #也转化为列
            dis = np.hstack((dis,d0)) #列与列合并成二维数组
        
        cost = dis.min(axis = 1).sum()
    
        medoids[-2] = [cost]
    
    def clustering(self,dist_mat, medoids):
        '''
        compute the belonging of each data point according to current medoids centers, and eucludiean distance.
        '''
        # pdb.set_trace()
        med_idx = medoids[-1]
        k = len(med_idx)
        dis = dist_mat[:,med_idx[0]]
        dis = dis[:,None] #将一维数组转化为二维数组的一列
    
        for i in range(1,k):
            d0 = dist_mat[:,med_idx[i]]
            d0 = d0[:,None] #也转化为列
            dis = np.hstack((dis,d0)) #列与列合并成二维数组
    
        best_med_it_belongs_to = dis.argmin(axis = 1)#按行
        for i in range(k):
            medoids[i] =where(best_med_it_belongs_to == i)        
        
    def kmedoids(self, dist_mat, k):
        '''
        given the data and # of clusters, compute the best clustering based on the algorithm provided in wikipedia: 
        google pam algorithm.
        '''
        cur_medoids = {}
        cur_medoids[-1] = range(k)
        self.clustering(dist_mat, cur_medoids)
        self.total_cost(dist_mat, cur_medoids)
        old_medoids = {}
        old_medoids[-1] = []
    
        iter_counter = 1
        # stop if not improvement.
        while not set(old_medoids[-1]) == set(cur_medoids[-1]):
            print 'iteration couter:' , iter_counter
            iter_counter = iter_counter + 1
            best_medoids = copy.deepcopy(cur_medoids) #深复制 新建一个变量
            old_medoids = copy.deepcopy(cur_medoids)
            # pdb.set_trace()
            # iterate over all medoids and non-medoids
            for i in range(len(dist_mat)):
                for j in range(k):
                    if not i ==j :
                        # swap only a pair
                        tmp_medoids = copy.deepcopy(cur_medoids)
                        tmp_medoids[-1][j] = i

                        self.clustering(dist_mat, tmp_medoids)
                        self.total_cost(dist_mat, tmp_medoids)
                        # pick out the best configuration.
                        if( best_medoids[-2] > tmp_medoids[-2]):
                            best_medoids = copy.deepcopy(tmp_medoids)
            cur_medoids = copy.deepcopy(best_medoids)
            print 'current total cost is ', cur_medoids[-2]
        return cur_medoids

class SecondPhaseFunction:
    '''
    第二阶段所使用的函数
    
    '''
    def kmedoid(self,dist_mat,k):
        'dist_mat-距离矩阵，k－簇的个数,返回－分群结果'
        km = KMEDOIDS()
        medoids = km.kmedoids(dist_mat, k)
        return medoids
    
    def hieracy(self,dist_mat,indivdualLabel,linkage,title):
        'dist_mat-距离矩阵,indivdualLabel-画图标识,linkage,title- 图的名称'
        ytdist=[]
        for i in range(len(dist_mat)):
            for j in range(i+1,len(dist_mat)):
                ytdist.append(dist_mat[i,j])
                
        Z = hierarchy.linkage(ytdist, linkage)#complete average single
#         print 'Z:',Z
        plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
        plt.rcParams['font.sans-serif']=['STHeiti']#用来画图正常显示中文
        plt.figure()
        plt.title(title)
#         print self.indivdualLabel
        dn = hierarchy.dendrogram(Z,labels=indivdualLabel)

