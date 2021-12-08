#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pymssql
import dbfread
import pandas as pd
import numpy as np
import sqlalchemy
import pyodbc
import time
import datetime

class SQLServer:
    def __init__(self,server,user,password,database):
        self.server = server
        self.user = user
        self.password = password
        self.database = database

    def __GetConnect(self):
        if not self.database:
            raise(NameError,"没有设置数据库信息")
        self.conn = pymssql.connect(server=self.server,user=self.user,password=self.password,database=self.database)
        cur = self.conn.cursor()
        if not cur:
            raise(NameError,"连接数据库失败")
        else:
            return self.conn

    def ExecQuery(self,sql):
        try:
            conn = self.__GetConnect()
            df=pd.read_sql(sql,conn)
            conn.close()
            return df
        except:
            conn.rollback()
            print('查询失败')
        return False
    
    def ExecOperate(self,sql):
        try:
            conn = self.__GetConnect()
            cur = conn.cursor()
            cur.execute(sql)
            conn.commit()
            conn.close()
        except:
            conn.rollback()
            print('操作失败')
            raise(NameError,"操作失败")
            return False
        return True
    
    def ExecOperateMany(self,sql,self_list):
        try:
            conn = self.__GetConnect()
            cur = conn.cursor()
            cur.executemany(sql,self_list)
            conn.commit()
            conn.close()
        except:
            conn.rollback()
            print('操作失败')
            raise(NameError,"操作失败")
            return False
        return True
    
    def ExecTable(self,df,table,my_engine):
        try:
            conn = self.__GetConnect()
            cur = conn.cursor()
            sql="""
                IF OBJECT_ID('"""+table+"""','U') IS NOT NULL
                    DROP TABLE """+table+"""
            """
            cur.execute(sql)
            conn.commit()
            engine=sqlalchemy.create_engine("mssql+pyodbc://"+self.user+":"+self.password+my_engine)
            df.to_sql(table,engine)
            print('建表成功')
            conn.close()
        except:
            conn.rollback()
            print('建表失败')
            raise(NameError,"建表失败")
            return False
        return True

#四个参数，主机地址，用户，密码，数据库
try:
    msg = SQLServer(server="127.0.0.1",user="cfydzl",password="123506467",database="ccf")
    print('数据库连接成功')
except:
    print('数据库连接失败')


# In[28]:


#贝叶斯推荐算法(方案一)
# user_id_max=5W
user_id=5000
# item_id_max=1W
item_id=3000
# 二分查找
def binarySearch (arr, l, r, x): 
    while r>=l:
        mid = int(l + (r - l)/2)
        if int(arr[mid]) == x: 
            return mid
        elif int(arr[mid]) > x: 
            r=mid-1
        else:
            l=mid+1
    return -1

# 数据获取
def getdata():
    train_dataset = [[] for _ in range(user_id)]
    sql="""select *from [dbo].[train_dataset] where user_id <"""+str(user_id)+""" and item_id <"""+str(item_id)+""" order by user_id asc,item_id asc"""
    record_data=msg.ExecQuery(sql)
    for index,row in record_data.iterrows():
        train_dataset[row['user_id']].append(row['item_id'])
    return train_dataset

# 数据输入
def inputdata(buy_book):
    train_dataset=getdata()
    choose_dataset=[]
    A_dataset=0
    train_dataset_len=len(train_dataset)
    for i in range(train_dataset_len):
        flag=1
        for j in buy_book:
            if binarySearch(train_dataset[i], 0, len(train_dataset[i])-1, int(j))==-1:
                flag=0
                break
        if flag==1:
            A_dataset=A_dataset+1
            choose_dataset.append(i)
    A_dataset=A_dataset/user_id
    return choose_dataset,A_dataset,buy_book,train_dataset

# 结果预测
def outputdata(buy_book):
    AandB_dataset = [0] *item_id
    AtoB_dataset={}
    choose_dataset,A_dataset,buy_book,train_dataset=inputdata(buy_book)
    for i in choose_dataset:
        for j in range(item_id):
            if binarySearch(train_dataset[i], 0, len(train_dataset[i])-1, j)!=-1:
                AandB_dataset[j]=AandB_dataset[j]+1
    AandB_dataset_len=len(AandB_dataset)
    for i in range(AandB_dataset_len):
        if str(i) not in buy_book:
            if user_id==0 or A_dataset==0:
                AtoB_dataset.update({i:0})
            else:
                AtoB_dataset.update({i:AandB_dataset[i]/user_id/A_dataset})
    return sorted(AtoB_dataset.items(), key=lambda item:item[1], reverse=True)

# 预测保存
def savedata():
    buy_book=input().split()
    result=outputdata(buy_book)
    dataframe = pd.DataFrame(result)
    dataframe.to_excel(r'E:\CCF\结果预测\result_bys_1.xls')
    print("结果保存完成")

before_time = datetime.datetime.now()
savedata()
print('耗时：'+str(datetime.datetime.now()-before_time))


# In[29]:


# 贝叶斯推荐算法(方案二)
# user_id_max=5W
user_id=5000
# item_id_max=1W
item_id=3000
# 数据获取

def next_function(train_dataset,result):
    for i in train_dataset:
        result[i]=result[i]+1
    return result

# 获取数据并处理
def getdata(buy_book,result):
    train_dataset = []
    sql="""select *from [dbo].[train_dataset] where user_id <"""+str(user_id)+""" and item_id <"""+str(item_id)+""" order by user_id asc,item_id asc"""
    record_data=msg.ExecQuery(sql)
    now_user_id=-1
    now_book=0
    max_buy_len=len(buy_book)
    for index,row in record_data.iterrows():
        if now_user_id!=row['user_id']:
            if now_book==max_buy_len:
                result=next_function(train_dataset,result)
            train_dataset.clear()
            now_book=0
            now_user_id=row['user_id']
        train_dataset.append(row['item_id'])
        if now_book!=max_buy_len and int(buy_book[now_book])==row['item_id'] and now_user_id==row['user_id']:
            now_book=now_book+1
    if max_buy_len==now_book-1:
        result=next_function(train_dataset,result)
    return result

# 预测计算
def outputdata(buy_book,result):
    predict_result={}
    result_len=len(result)
    for i in range(result_len):
        if str(i) not in buy_book:
            predict_result[i]=(result[i]/item_id)/(result[int(buy_book[0])]/item_id)
    return sorted(predict_result.items(), key=lambda item:item[1], reverse=True)

def savedata(predict_result):
    dataframe = pd.DataFrame(predict_result)
    dataframe.to_excel(r'E:\CCF\结果预测\result_bys_2.xls')
    print("结果保存完成")
# 数据输入
def inputdata():
    result=[0]*item_id
    buy_book=input().split()
    result=getdata(buy_book,result)
    predict_result=outputdata(buy_book,result)
    savedata(predict_result)
before_time = datetime.datetime.now()
inputdata()
print('耗时：'+str(datetime.datetime.now()-before_time))


# In[30]:


# 频繁项集推荐算法（贝叶斯优化预处理数据）
# user_id_max=5W
user_id_max=5000
# item_id_max=1W
item_id_max=3000

# 数据获取
def getdata():
    train_dataset = [[] for _ in range(user_id_max)]
    sql="""select *from [dbo].[train_dataset] where user_id <"""+str(user_id_max)+""" and item_id <"""+str(item_id_max)+""" order by user_id asc,item_id asc"""
    record_data=msg.ExecQuery(sql)
    for index,row in record_data.iterrows():
        train_dataset[row['user_id']].append(row['item_id'])
    return train_dataset

# 初始候选集
def createH(train_dataset):
    H = []
    for i in range(item_id_max):
        H.append([i])
    return list(map(frozenset,H))

#生成过滤集
def createG(train_dataset,H,minsupport):
    now_bys={}
    H_num={}
    G=[]
    for user_id_record in train_dataset:
        for item_id in H:
            if item_id.issubset(user_id_record):
                if item_id not in H_num:
                    H_num[item_id]=1
                else:
                    H_num[item_id]+=1
    for item_id in H_num:
        if H_num[item_id]/user_id_max>=minsupport:
            G.append(item_id)
        now_bys[str(item_id)]=H_num[item_id]/user_id_max
    return G,now_bys

# 递推候选集
def create_newH(H,k):
    new_dict={}
    H_len=len(H) 
    for i in range(H_len):
        for j in range(i+1,H_len):
            new_id=H[i]|H[j]
            if len(new_id)==k:
                if new_id not in new_dict:
                    new_dict[new_id]=1
    return list(new_dict)

# apriori算法
def apriori(train_dataset, minsupport = 0.1):
    data=list(map(set,train_dataset))
    sum_bys={}
    H=createH(train_dataset)
    G,sum_bys=createG(data,H,minsupport)
    G_sum=[G]
    k=2
    while len(G_sum[k-2])>0:
        H=create_newH(G_sum[k-2],k)
        G,now_bys=createG(data,H,minsupport)
        sum_bys.update(now_bys)
        G_sum.append(G)
        k=k+1
    return  sum_bys

# 预处理保存
def savedata():
    train_dataset = getdata()
    result=apriori(train_dataset)
    dataframe =pd.DataFrame(list(result.items()))
    dataframe.to_excel(r'E:\CCF\结果预测\result_apriori.xls')
    print("结果保存完成")
before_time = datetime.datetime.now()
savedata()
print('耗时：'+str(datetime.datetime.now()-before_time))


# In[51]:


# 基于多维线性代数的cos距离算法
# user_id_max=5W
user_id_max=5000
# item_id_max=1W
item_id_max=3000

# 数据获取
def getdata():
    train_dataset = [[] for _ in range(item_id_max)]
    sql="""select *from [dbo].[train_dataset] where user_id <"""+str(user_id_max)+""" and item_id <"""+str(item_id_max)+""" order by user_id asc,item_id asc"""
    df=msg.ExecQuery(sql)
    for index,row in df.iterrows():
        train_dataset[int(row['item_id'])].append(int(row['user_id']))
    return train_dataset

# 矩阵初始化
def get_matrix():
    train_dataset=getdata()
    matrix_dataset = [[0]*item_id_max for _ in range(item_id_max)]
    for i in range(item_id_max):
        now_i_len=len(train_dataset[i])
        matrix_dataset[i][i]=1
        for j in range(i+1,item_id_max):
            now_dis=now_i_len**0.5+len(train_dataset[j])**0.5
            if now_dis==0:
                matrix_dataset[i][j]=0
                matrix_dataset[j][i]=0
            else:
                now_same_dataset=set(train_dataset[i])&set(train_dataset[j])
                matrix_dataset[i][j]=len(now_same_dataset)/now_dis
                matrix_dataset[j][i]=len(now_same_dataset)/now_dis
    return matrix_dataset

# 预测函数
def predict_now(now_user_record,matrix_dataset):
    now_user_predict={}
    now_user_result=[]
    for i in range(item_id_max):
        now_score=0
        if i in now_user_record:
            continue
        for j in now_user_record:
            now_score=now_score+matrix_dataset[i][j]
        if len(now_user_predict)<10:
            now_user_predict[i]=now_score
        else:
            now_min_dict_key=min(now_user_predict,key=now_user_predict.get)
            if now_user_predict[now_min_dict_key]<now_score:
                now_user_predict.pop(now_min_dict_key)
                now_user_predict[i]=now_score
    now_user_predict=sorted(now_user_predict.items(), key=lambda item:item[1], reverse=True)
    for i in now_user_predict:
        now_user_result.append(i[0])
    return now_user_result

def predict_end(now_user_id,now_user_record,matrix_dataset,submission_csv):
    now_user_predict_list=predict_now(now_user_record,matrix_dataset)
    for j in now_user_predict_list:
        submission_csv.append([now_user_id,j])
    return submission_csv

# 结果预测
def predict_all(submission_csv):
    matrix_dataset=get_matrix()
    sql="""select *from [dbo].[train_dataset] where user_id <"""+str(user_id_max)+""" and item_id <"""+str(item_id_max)+""" order by user_id asc,item_id asc"""
    all_user_data=msg.ExecQuery(sql)
    now_user_id=0
    now_user_record=[]
    for index,row in all_user_data.iterrows():
        if row['user_id']==now_user_id:
            now_user_record.append(row['item_id'])
        else:
            now_user_predict_list=predict_now(now_user_record,matrix_dataset)
            now_user_record.clear()
            for j in now_user_predict_list:
                submission_csv.append([row['user_id']-1,j])
            now_user_id=row['user_id']
            now_user_record.append(row['item_id'])
    submission_csv=predict_end(now_user_id,now_user_record,matrix_dataset,submission_csv)
    return submission_csv

# 运行函数
def savedata():
    submission_csv=[]
    submission_csv=predict_all(submission_csv)
    df = pd.DataFrame(submission_csv)
    df.to_csv(r'E:\CCF\结果预测\submission.csv')
    print("预测完成")
before_time = datetime.datetime.now()
savedata()
print('耗时：'+str(datetime.datetime.now()-before_time))


# In[172]:


#基于贝叶斯过滤的物品协同预测算法
# user_id_max=5W
user_id_max=5000
# item_id_max=1W
item_id_max=3000

# 数据获取
def getdata():
    base_item_dataset = [[] for _ in range(item_id_max)]
    base_user_dataset = [[] for _ in range(user_id_max)]
    sql="""select *from [dbo].[train_dataset] where user_id <"""+str(user_id_max)+""" and item_id <"""+str(item_id_max)+""" order by user_id asc,item_id asc"""
    df=msg.ExecQuery(sql)
    for index,row in df.iterrows():
        x=row['item_id']
        y=row['user_id']
        base_item_dataset[int(x)].append(int(y))
        base_user_dataset[int(y)].append(int(x))
    return base_item_dataset,base_user_dataset

# 物品权值矩阵初始化
def get_matrix():
    base_item_dataset,base_user_dataset=getdata()
    item_value_dataset = [[0]*item_id_max for _ in range(item_id_max)]
    for i in range(item_id_max):
        now_i_len=len(base_item_dataset[i])
        item_value_dataset[i][i]=1
        for j in range(i+1,item_id_max):
            now_dis=now_i_len**0.5+len(base_item_dataset[j])**0.5
            if now_dis==0:
                item_value_dataset[i][j]=0
                item_value_dataset[j][i]=0
            else:
                now_same_dataset=set(base_item_dataset[i])&set(base_item_dataset[j])
                item_value_dataset[i][j]=len(now_same_dataset)/now_dis
                item_value_dataset[j][i]=len(now_same_dataset)/now_dis
    return base_item_dataset,base_user_dataset,item_value_dataset

#贝叶斯过滤数据统计
def bys_filter(bys_dataset):
    bys_filter_dataset=[[0]*item_id_max for _ in range(item_id_max)]
    sum_num=[0]*item_id_max
    bys_dataset_len=len(bys_dataset)
    for i in range(bys_dataset_len):
        now_len=len(bys_dataset[i])
        for j in range(now_len):
            sum_num[bys_dataset[i][j]]+=1
            bys_filter_dataset[bys_dataset[i][j]][bys_dataset[i][j]]+=1
            for k in range(j+1,now_len):
                bys_filter_dataset[bys_dataset[i][j]][bys_dataset[i][k]]+=1
                bys_filter_dataset[bys_dataset[i][k]][bys_dataset[i][j]]+=1
    for i in range(item_id_max):
        if sum_num[i]==0:
            bys_filter_dataset[i][i]=0
        else:
            bys_filter_dataset[i][i]=bys_filter_dataset[i][i]/sum_num[i]
        for j in range(i+1,item_id_max):
            if sum_num[i]==0:
                bys_filter_dataset[j][i]=0
            else:
                bys_filter_dataset[j][i]=bys_filter_dataset[j][i]/sum_num[i]
            if sum_num[j]==0:
                bys_filter_dataset[i][j]=0
            else:
                bys_filter_dataset[i][j]=bys_filter_dataset[i][j]/sum_num[j]
    return bys_filter_dataset

#贝叶斯过滤矩阵运算
def matrix_fun(user_items_dataset,bys_filter_dataset,base_user_dataset):
    for i in range(user_id_max):
        now_len=len(base_user_dataset[i])
        for j in range(item_id_max):
            if user_items_dataset[i][j]==1:
                continue
            value=0
            num=0
            for k in range(now_len):
                value+=bys_filter_dataset[j][base_user_dataset[i][k]]
                num+=1
            if num==0:
                user_items_dataset[i][j]=0
            else:
                user_items_dataset[i][j]=value/num
    return user_items_dataset
                
# 用户物品矩阵
def user_items_build(base_user_dataset,bys_filter_dataset):
    user_items_dataset=[[0]*item_id_max for _ in range(user_id_max)]
    for i in range(user_id_max):
        base_user_len=len(base_user_dataset[i])
        for j in range(base_user_len):
            user_items_dataset[i][base_user_dataset[i][j]]=1
    user_items_dataset=matrix_fun(user_items_dataset,bys_filter_dataset,base_user_dataset)
    return user_items_dataset
    
#用户推荐预测
def predict_now(result_matrix,submission_csv,user_items_dataset):
    for i in range(user_id_max):
        now_user_predict={}
        now_user_result=[]
        for j in range(item_id_max):
            if user_items_dataset[i][j]==1:
                continue
            if len(now_user_predict)<10:
                now_user_predict[j]=result_matrix[i][j]
            else:
                now_min_dict_key=min(now_user_predict,key=now_user_predict.get)
                if now_user_predict[now_min_dict_key]<result_matrix[i][j]:
                    now_user_predict.pop(now_min_dict_key)
                    now_user_predict[j]=result_matrix[i][j]  
        now_user_predict=sorted(now_user_predict.items(), key=lambda item:item[1], reverse=True)
        for j in now_user_predict:
            submission_csv.append([i,j[0]])
    return submission_csv
# 结果预测
def predict_all(submission_csv):
    base_item_dataset,base_user_dataset,item_value_dataset=get_matrix()
    bys_filter_dataset=bys_filter(base_user_dataset)
    user_items_dataset=user_items_build(base_user_dataset,bys_filter_dataset)
    result_matrix=np.dot(user_items_dataset,item_value_dataset)
    submission_csv=predict_now(result_matrix,submission_csv,user_items_dataset)
    return submission_csv
# 运行函数
def savedata():
    submission_csv=[]
    submission_csv=predict_all(submission_csv)
    df = pd.DataFrame(submission_csv)
    df.to_csv(r'E:\CCF\结果预测\result_bys_base_item.csv')
    print("预测完成")
    
before_time = datetime.datetime.now()
savedata()
print('耗时：'+str(datetime.datetime.now()-before_time))


# In[ ]:




