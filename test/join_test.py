import pandas as pd

df1=pd.DataFrame({'key':['a','b','b','d'],'data1':range(4)})
df2=pd.DataFrame({'key':['a','b','c'],'data2':range(3)})
print(df1)
print(df2)
print(pd.merge(df1,df2))