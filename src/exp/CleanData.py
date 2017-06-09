#coding:utf-8
'''
Created on 2017年2月15日

@author: whl
'''
from  Util import Util  #from Util模块，import util模块中的util类

class DataClean:
    '数据清理'
    # 定义没用的列名，此属性为类变量，相当于java的静态变量,访问权限为public
    usernessColumns=['N/A','Elevation','Phase','Marker','Env. Temp.','Analyz. Temp.',
                 'Analyz. Press.','Env. Press.','Batteries','Aux 1A','Aux 2A','Aux 3A',
                 'Aux 4A','Aux 1B','Bias Flow','Steady State','PROg','PROkc','PRO%',
                 'mark Speed','mark Dist.','ST I','ST II','ST III','ST aVR','ST aVL',
                 'ST aVF','ST V1','ST V2','ST V3','ST V4','ST V5','ST V6','S I',
                 'S II','S III','S aVR','S aVL','S aVF','S V1','S V2','S V3',
                 'S V4','S V5','S V6','P Syst','P Diast','Symptom','DP','Stage',
                'RR','Phase time','Vt/FVC','Long','Lat','Alt','GPS Speed','GPS Dist.',
                'O2 Cost','IC','Step','User 1','User 2','User 3','SpO2','FiO2','FiCO2','FeCO2','FeO2',
                'FATkc','FATg','FAT%','CHOkc','CHOg','CHO%','EEh','EEkc']
    
    inputDataPath='../originalData/'
    outputDataPath='../cleaneddata/'
    # 删除没用的列
    '静态方法可以没有self参数，可以访问类变量，不能访问实例变量'
    @staticmethod
    def deleteUsenessColumns(data,delColumnList):
        print '删除属性个数：',len(delColumnList)
        return data.drop(delColumnList,axis=1)
    
    '删除原始数据的周围info等额外数据'
    def delExtraData(self,data,outputPath):      
        data=data.iloc[7:,10:] #除去最前面的9列和前5行
        data.index=[i for i in range(len(data))] # 从0开始重新索引
        print '总的属性个数：',len(data.columns)
        data=DataClean.deleteUsenessColumns(data,DataClean.usernessColumns)  #删除没有的列
        print '剩余属性个数：',len(data.columns)
        data.to_excel(outputPath)
        
if __name__=='__main__':
    print '在此模块中运行'
    cleanData=DataClean()
    fileList=Util.readAllFileNameFromDir(cleanData.inputDataPath) #先调用读取文件夹下面所有的文件名
    # print len(fileList)
    for i in range(len(fileList)):
        if fileList[i]=='.DS_Store':
            continue;
        data=Util.readDataFromExcel(cleanData.inputDataPath+fileList[i])
        cleanData.delExtraData(data,cleanData.outputDataPath+fileList[i][:3]+'.xls') #循环处理
        
        
