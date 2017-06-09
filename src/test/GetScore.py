#coding:utf-8
import os

def readAllFileNameFromDir(dataPath):
        filenameList=os.listdir(dataPath)
        #     print (filenameList)
        return filenameList
    
if __name__=='__main__':
    
    filenameList=readAllFileNameFromDir('')