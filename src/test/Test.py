#coding:utf-8
import pandas as pd
list1=[[1,2],[2,1],[3,4],[4,3]]
list2=[[3,4],[4,3],[1,2],[2,1]]

df1=pd.DataFrame(list1)
df2=pd.DataFrame(list2)

print (df1-df2)
print (df1-df2)*(df1-df2)
print ((df1-df2)*(df1-df2)).sum().sum()