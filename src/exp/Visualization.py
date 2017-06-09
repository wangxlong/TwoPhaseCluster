#coding:utf-8
'''
 use to visulate data
 
'''
import matplotlib.pyplot as plt
import numpy as np

class SequencePlot():
    '''
    画序列图
    '''
    #画实线，平行于横坐标
    def plotSolidState(self,ax,x,y):
        ax.plot(x,y, linestyle='-', color='g', linewidth=0.7)
    
    #画虚线，垂直于横坐标
    def plotDottedState(self,ax,x,y):
        ax.plot(x,y, linestyle='--', color='r', linewidth=0.3)
        
    #画整条序列图
    def plotHidden_State(self,hidden_state,title):
        'hidden_state- 一条状态序列，title－图的名称'
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
                self.plotSolidState(ax, x, y)
                self.plotDottedState(ax, [i,i],[prestate,nextstate])
                x=[]
                y=[]
                x.append(i)
                y.append(nextstate)
                prestate=nextstate
                continue
        x.append(hidden_state_length)
        y.append(hidden_state[hidden_state_length-1])
        self.plotSolidState(ax, x, y)  
        ax.set_title(title)      
        plt.xticks(np.arange(0,hidden_state_length+1,hidden_state_length/15)) 
        
        
        