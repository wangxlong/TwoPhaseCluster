#coding:utf-8
'''
evaluation index

'''
class SpiValue():
    '''
     spi_value评价指标
    '''
    def __init__(self,observation,state,k):
        self.observation=observation #原始序列
        self.state=state #状态序列
        self.k=k #簇的个数
        self.size=len(self.state) #序列长度
    
    @staticmethod
    def compute_encludean(v1,v2):
        dist=0
        for i in range(len(v1)):
            dist+=pow(abs(v1[i]-v2[i]),2)   
        return dist 
       
    def compute_connectnumber(self,c1,c2):
        '计算c1和c2的联通度'
        sumconnc=0
        for i in range(self.size-1):
            if self.state[i]==c1 and self.state[i+1]==c2:
                sumconnc+=1
        return sumconnc
    
    def compute_diam(self,c):
        '计算簇c的半径'
        c_element=[]
        for i in range(self.size):
            if c==self.size[i]:
                c_element.append(self.observation[i])#observation[i] is multi-dimension
        
        max_dist=0
        for i in range(len(c_element)-1):
            for j in range(i+1,len(c_element)):
                curr_dist=SpiValue.compute_encludean(c_element[i],c_element[j])
                if max_dist<curr_dist:
                    max_dist=curr_dist
        return max_dist           
            
    def compute_dist_conn(self,c1,c2):
        '计算c1和c2的联通距离'
        dist_connc=0
        for i in range(self.size-1):
            if self.state[i]==c1 and self.state[i+1]==c2:
                dist_connc+=SpiValue.compute_encludean(self.observation[i], self.observation[i+1])
        return dist_connc
            
    def compute_avyconnlen(self,c1,c2):
        '计算c1和c2的平均联通距离'
        if self.compute_connectnumber(c1, c2)==0:
            return 0
        avy_conn=(self.compute_dist_conn(c1,c2)*1.0)/(self.compute_connectnumber(c1, c2)*1.0)
        return avy_conn
    
    def get_C(self):
        '将observation中的每个类别的数据分开，分别存在各自的数组中，则C是一个二维数组'
        state_C=[]
        observation_C=[]
        for i in range(self.k):
            state_C.append([])    
            observation_C.append([])       
        for i in range(self.size):
            state_C[self.state[i]].append(self.state[i])
            observation_C[self.state[i]].append(self.observation[i])
        return state_C,observation_C
            
    def intra_coeff(self):
        '计算 intra_coeff'
        intra_coeff=0.0
        for i in range(self.k):
            intra_coeff=intra_coeff+(self.compute_connectnumber(i, i)*1.0)/(self.compute_dist_conn(i,i)*1.0)
        return intra_coeff
        
    def inter_coeff(self):
        '计算 inter_coeff'
        inter_coeff=0
        for i in range(self.k):
            for j in range(self.k):
                if i==j:
                    continue;
                if self.compute_avyconnlen(i,j)==0:
                    continue
                inter_coeff+=(self.compute_connectnumber(i, j)*1.0)/(self.compute_avyconnlen(i,j)*1.0)
        return inter_coeff
    
    def fit(self):
        '得到spi_value的值'
        intra_coeffs=self.intra_coeff()
#         print 'intra_coeffs',intra_coeffs
        inter_coeffs=self.inter_coeff()
#         print 'intra_coeffs:',intra_coeffs
#         print 'inter_coeffs:',inter_coeffs
        spi_value=intra_coeffs/inter_coeffs
        return spi_value

class Evaluationfunction():
    '''
    评价指标 
    '''
    def spivalue(self,observation,state,k):
        ' observation-观察序列,state-观察数据对应的状态,k-状态个数，返回－spivalue数值'
        spiValue=SpiValue(observation,state,k)
        return spiValue.fit()
        
