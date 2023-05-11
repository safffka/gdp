import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

salary=pd.read_excel('salary.xlsx')
#drop first 20 rows
salary=salary.drop(salary.index[0:18])
#drop last 3 rows
salary=salary.drop(salary.index[-7:])
df=salary[['Unnamed: 2','Unnamed: 3','Unnamed: 4','Unnamed: 5']]
df1=df['Unnamed: 2']
df1=df1.to_numpy()
df2=df['Unnamed: 3']
df2=df2.to_numpy()
df3=df['Unnamed: 4']
df3=df3.to_numpy()
df4=df['Unnamed: 5']
df4=df4.to_numpy()
avg_salary=[]
for i in range(len(df1)):
    avg_salary.append(df1[i])
    avg_salary.append(df2[i])
    avg_salary.append(df3[i])
    avg_salary.append(df4[i])
#delete avg_salary[0]
avg_salary=avg_salary[1:]
#create dataframe with index=qurter and column=avg_salary
salary=pd.DataFrame(avg_salary,columns=['avg_salary'])
salary=salary['avg_salary'].to_numpy()
np.savez('salary.npz', salary=salary)

