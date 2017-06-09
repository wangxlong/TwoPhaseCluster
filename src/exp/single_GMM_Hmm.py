#coding:utf-8
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import  mixture
from numpy import mat ,zeros
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy import linalg
import matplotlib.pyplot as plt

# 将所有数据当作一条进行gmm_hmm
class single_gmm_hmm():
    
    @staticmethod
    def getStateTransMat(series,n):
        'series表示状态序列，n表示状态序列的个数，series取值范围为（0，1，2，……,n-2，n-1）'
        TransMat=mat(zeros((n,n)))
        pa=np.zeros( (n) )
        
        TransMatsum=np.zeros( (n) )
        pasum=np.zeros((n))
        size=len(series)
        for i in range(size-1):
            TransMat[series[i],series[i+1]]+=1
            TransMatsum[series[i]]+=1
            pasum[series[i]]+=1
#         print sumi
#         print TransMat
        pasum[series[size-1]]+=1
        for j in range(n):
            TransMat[j]=TransMat[j]/TransMatsum[j]
            pa[j]=pasum[j]/size
#         print TransMat
        return TransMat,pa
    
    
    def initialHidden_State(self,globalstatenumber,data):
        gmm = mixture.GaussianMixture(n_components=globalstatenumber,random_state=1, covariance_type='diag',max_iter=1000).fit(data)  #高斯混合处理

        labels=gmm.predict(data) #得到初始化序列        
        return labels,gmm.means_,gmm.covariances_
    
    def getFinalState(self,globalstatenumber,data,pa,transport,means,convars):
        model = GaussianHMM(n_components=globalstatenumber,n_iter=1000,covariance_type='diag',params='stcm', init_params='',random_state=1)
        model.startprob_=pa
        model.transmat_=transport
        model.means_=means
        model.covars_=convars
        #-----------
#         for i, con in enumerate(model.covars_):
#             if (not np.allclose(con, con.T) or np.any(linalg.eigvalsh(con) <= 0)):
#                 print 'is:',i
#             else:
#                 print 'not is:',i
        
#         print 'before_model.covars_:',model.covars_
        model.fit(data)
        hidden_states = model.predict(data)
#         print 'after_model.covars_:',model.covars_
        return hidden_states
     
    def hidden_state_output(self,hidden_states,outputfilename):
        userflag=pd.read_csv('First_Phase_Gmm_Data/userLabel.csv',header=None,names=['name','index'])
        maxLengthIs=np.max(userflag['index'])
#         print maxLengthIs
        beforeindex=lastindex=0
        clusterLabeldic={}
        for i in range(len(userflag)):          
            lastindex=lastindex+userflag['index'][i]
            everyclusterLabel=hidden_states[beforeindex:lastindex]
            beforeindex=beforeindex+userflag['index'][i]
            everyclusterLabel=everyclusterLabel.tolist()
            
            single_gmm_hmm.plotHidden_State(everyclusterLabel,userflag['name'][i])
#             print everyclusterLabel          
#             #为了写出excel，通过数据补齐，然后构造dataframe结构，然后写出excel            
            for j in range(maxLengthIs-len(everyclusterLabel)):#等长补齐
                everyclusterLabel.append(None)
            clusterLabeldic[userflag['name'][i]]=everyclusterLabel
#             print len(everyclusterLabel)
        clusterlabeldf=pd.DataFrame(clusterLabeldic)
        clusterlabeldf.to_csv('singleHmmData/'+outputfilename+'.csv')
        return everyclusterLabel
    
    @staticmethod
    #画实线，平行于横坐标
    def plotSolidState(ax,x,y):
        ax.plot(x,y, linestyle='-', color='g', linewidth=0.7)
    
    @staticmethod
    #画虚线，垂直于横坐标
    def plotDottedState(ax,x,y):
        ax.plot(x,y, linestyle='--', color='r', linewidth=0.3)
        
    #画整条序列图
    @staticmethod 
    def plotHidden_State(hidden_state,title):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        hidden_state_length=len(hidden_state)  
        x=[]
        y=[]
        prestate=hidden_state[0]
        x.append(0)
        y.append(prestate)
        for i in range(1,hidden_state_length):
            nextstate=hidden_state[i]
            if(prestate==nextstate):
                x.append(i)
                y.append(nextstate)
                prestate=nextstate
                continue
            if(prestate!=nextstate):
                x.append(i)
                y.append(prestate)
                single_gmm_hmm.plotSolidState(ax, x, y)
                single_gmm_hmm.plotDottedState(ax, [i,i],[prestate,nextstate] )
                x=[]
                y=[]
                x.append(i)
                y.append(nextstate)
                prestate=nextstate
                continue
        x.append(hidden_state_length)
        y.append(hidden_state[hidden_state_length-1])
        single_gmm_hmm.plotSolidState(ax, x, y)  
        ax.set_title(title)      
        plt.xticks(np.arange(0,hidden_state_length+1,hidden_state_length/15))  
       
if __name__=='__main__':
    print '直接运行'
    #数据加载
    filePath='First_Phase_Gmm_Data/firstPhaseInputData.xls'
    features=['Rf','VT','VE','VO2','VCO2','O2exp','CO2exp','VO2/Kg','R','HR','FetO2','FetCO2','Ti','Te','Ttot','Ti/Ttot',
              'IV','PetO2','PetCO2','PaCO2','PAO2','VD(phys)','VD/VT','EEm','EEbsa','EEkg','npRQ','t Rel',
              'METS','Qt','SV','predVO2','BR','EEtot']#特征列表
    data=pd.read_excel(filePath)#读取外存数据
    data=data[features] #特征抽取
    print len(data)
    data=data.dropna()
    print len(data)
#     print data
    data=np.array(data)
    data= StandardScaler().fit_transform(data)#0-1均值化！！！！！！！！！！！！！

    s_gmm_hmm=single_gmm_hmm()
    initial_hidden_states,mean,conv=s_gmm_hmm.initialHidden_State(data)
#     print 'means,conv:'
                       
    print 'initial_hidden_states:',initial_hidden_states
    
    transport,pa=single_gmm_hmm.getStateTransMat(initial_hidden_states,9)
    
    final_hidden_states=s_gmm_hmm.getFinalState(data,pa,transport,mean,conv)
    print 'final_hidden_states:',final_hidden_states
     
    s_gmm_hmm.hidden_state_output(initial_hidden_states,'initial_hidden_states')
    s_gmm_hmm.hidden_state_output(final_hidden_states,'final_hidden_states')

#     print everyclusterLabel
    single_gmm_hmm.plotHidden_State(final_hidden_states,'final_series')
    
    plt.show()
    
    