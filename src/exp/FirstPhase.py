#coding:utf-8
'''
 first phase code
'''
from  Util import Util
import random
import math
import copy
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import  mixture
from sklearn.cluster import KMeans
from single_GMM_Hmm import single_gmm_hmm
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA

class fuzzyCM():
    '''
    fuzzy cmeans方法
    '''
    
    maxInterator=1000
    epsilon=1.0e-6
    m=2
    k=3
    
    def __init__(self,k,m=2,maxIte=1000,ep=1e-6):
        self.k=k
        self.maxInterator=maxIte
        self.epsilon=ep
        self.m=m
    
    def initial_U(self,data,k,random_state):
        U = []
        random_seek=np.random.RandomState(random_state)
        for i in range(0,len(data)):
            current = []
            rand_sum = 0.0
            for j in range(0,k):
                dummy = random_seek.randint(1,int(1000))
                current.append(dummy)
                rand_sum += dummy
            for j in range(0,k):
                current[j] = current[j] / rand_sum
            U.append(current)
        return U
    
    def dictance_cal(self,point,center):
        if len(point) != len(center):
            return -1
        dummy = 0.0
        for i in range(0,len(point)):
            dummy += abs(point[i] - center[i]) ** 2
        return math.sqrt(dummy)
    
    def end_conditon(self,U,U_old):
        for i in range(0,len(U)):
            for j in range(0,len(U[0])):
                if abs(U[i][j] - U_old[i][j]) > self.epsilon :
                    return False
        return True
    
    def get_Label(self,U):
         
        label=[]
        for i in range(0,len(U)):
            maximum = max(U[i])
            for j in range(0,len(U[0])):
                if U[i][j] == maximum:
                    label.append(j)            
        return label
    
    def fuzzy_cm(self,data,random_state):
#         print "begin clustering"
        ## initialise the U matrix:
        U = self.initial_U(data, self.k,random_state)
        #print_matrix(U)
        #initilise the loop
        while (True):
            #create a copy of it, to check the end conditions
            U_old = copy.deepcopy(U)
            # cluster center vector
            C = []
            for j in range(0,self.k):
                current_cluster_center = []
                for i in range(0,len(data[0])): #this is the number of dimensions
                    dummy_sum_num = 0.0
                    dummy_sum_dum = 0.0
                    for k in range(0,len(data)):
                        dummy_sum_num += (U[k][j] ** self.m) * data[k][i]
                        dummy_sum_dum += (U[k][j] ** self.m)
                    current_cluster_center.append(dummy_sum_num/dummy_sum_dum)
                C.append(current_cluster_center)
    
            #creating a distance vector, useful in calculating the U matrix.
    
            distance_matrix =[]
            for i in range(0,len(data)):
                current = []
                for j in range(0,self.k):
                    current.append(self.dictance_cal(data[i], C[j]))
                distance_matrix.append(current)
    
            # update U vector
            for j in range(0, self.k):    
                for i in range(0, len(data)):
                    dummy = 0.0
                    for k in range(0,self.k):
                        dummy += (distance_matrix[i][j]/ distance_matrix[i][k]) ** (2/(self.m-1))
                    U[i][j] = 1 / dummy
    
            if self.end_conditon(U,U_old):
#                 print "finished clustering"
                break
        label = self.get_Label(U)
        return np.array(label),U
    
class PAA():
    
    '''
     paa 方法
     
    '''
    
    def readFileToDf(self,path):#path='dealeddata/'
        '读取文件，每个文件封装为一个df，返回df集合'
        a = [] 
        fileList=Util.readAllFileNameFromDir(path)
        individul=[]
        for i in range(len(fileList)):
            df=pd.read_excel(path+fileList[i],encoding = 'utf-8')
            a.append(df)
            individul.append(fileList[i][:-4])
        return a,individul
    
    def t_traversal(self,factors,a_,i,l): 
        '删除多余的记录,factors表示每个速度段最后取的记录数,a_表示df集合，i表示从第几个记录开始' 
        j = i+1
        while(a_['Speed'][i] == a_['Speed'][j]):
            j = j+1
            if (j == l):
                break
        global k
        d = (j-i)%factors[k] 
        d_list = []
        for p in range (i,i+d):
            d_list.append(p)
        a_.drop(d_list,axis = 0,inplace = True)
        k = k+1
        #print 'index',i,'- index',j-1-d,'\t length:',j-i-d,'\t Speed Value:',a_speed[i]
        if (j < l):
            self.t_traversal(factors,a_, j,l)
           
    def compute_average(self,b_,l,factors):
        '计算得到最后的数据'
        ave_ = []
        n = 0
        i = 0
        while(i+1 < l):
            j = i+1
            while(b_['Speed'][i] == b_['Speed'][j]):
                j = j+1
                if (j == l):
                    break
            d = (j-i)/factors[n]
            k = i
            while(k < j):
                s = (b_.ix[k:k+d-1].sum())/d
                ave_.append(s)
                k = k+d
            i = j
            n = n+1
        return ave_
       
    def paa(self,path,factors):#path='dealeddata/'
        
        a_,individul=self.readFileToDf(path)
        '''k代表第几个参数值'''
        global k 
        k = 0
        b = []
        for i in range(len(a_)):
            L = len(a_[i]['Speed'])
            self.t_traversal(factors, a_[i], 0,L) 
            '''索引重置即删除值为空的记录'''
            b.append(a_[i].reset_index(drop = True))
            k = 0
        
        av_ = []
        for i in range (len(b)):
            av_.append(pd.DataFrame(self.compute_average(b[i],len(b[i]),factors)))
        #print b[0].ix[0],'\n',b[0].ix[1],'\n',b[0].ix[0:1].sum() 
        return av_  ,individul   

class FirstPhaseFunction:
    '''
    第一阶段所使用的方法函数,
    1、序列：
        输入：参数X表示观察数据
        输出：观察序列对应的状态序列
    2、paa:
        输入：path－数据的目录，piecewise－每个速度区间的段数
        输出：paa后的数据
    
    '''
    def kmeansSe(self,globalstatenumber,X):
        '使用kmeans产生序列'
        k_means=KMeans(n_clusters=globalstatenumber,max_iter=1000,random_state=1).fit(X)
        labels=k_means.labels_
        return labels
        
    def fcmeansSe(self,globalstatenumber,X):
        '使用fcmeans产生序列'
        fuzzy_cmeans_label,final_U= fuzzyCM(k=globalstatenumber,m=2).fuzzy_cm(X,1000)
        return fuzzy_cmeans_label
    
    def GMMSe(self,globalstatenumber,X):
        '使用GMM产生序列'
        gmm = mixture.GaussianMixture(n_components=globalstatenumber, covariance_type='diag',
                              max_iter=1000,random_state=1).fit(X)
        labels=gmm.predict(X)
        return labels
        
    def HMMSe(self,globalstatenumber,X):
        '使用hmm产生序列'
        model = GaussianHMM(n_components=globalstatenumber,n_iter=1000,random_state=1,covariance_type='diag',params='stcm', init_params='stcm')
        model.fit(X)
        hidden_states = model.predict(X)
        return hidden_states
        
    def gmmHmmSe(self,globalstatenumber,X):
        '使用gmm预处理产生序列'
        s_gmm_hmm=single_gmm_hmm()
        initial_hidden_states,mean,conv=s_gmm_hmm.initialHidden_State(globalstatenumber,X)
#     print 'means,conv:'                      
#         print 'initial_hidden_states:',initial_hidden_states
        transport,pa=single_gmm_hmm.getStateTransMat(initial_hidden_states,globalstatenumber)
        final_hidden_states=s_gmm_hmm.getFinalState(globalstatenumber,X,pa,transport,mean,conv)
        return final_hidden_states
    
    def paadeal(self,path,piecewise):
        'paa处理'
        pAA=PAA()
        data,individul=pAA.paa(path, piecewise)
        return data,individul
    
    def pcadeal(self,data,individul):
        'pca 处理,data-dataframe数据集合,返回pca处理后的学生数据列表、对应的权重列表以及个体标识 '
        pcadata=[]
        weightlist=[]
        for i in range(len(data)):
            pca=PCA()
            dealdata=pca.fit_transform(data[i])
            weightlist.append(pca.explained_variance_ratio_)
            pcadata.append(dealdata)
        
        return pcadata,weightlist,individul
            
        
if __name__=='__main__':
    
    print 'direct run'
    
    
    
    
    
    