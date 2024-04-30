# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 12:34:18 2024

@author: iyaku
"""
import numpy as np
import pandas as pd
df = pd.read_csv(r"C:\Users\iyaku\OneDrive\Masaüstü\PythonTrain\PyRecs/movie_genres.dat", delimiter='\t', encoding='ISO-8859-1') 
movi = df['movieID'].to_numpy()
mis=np.unique(movi)
train_ratings= pd.read_csv(r"C:\Users\iyaku\Desktop\HybridCF/train_ratings.txt", delimiter='\t') 
test_ratings= pd.read_csv(r"C:\Users\iyaku\Desktop\HybridCF/test_ratings.txt", delimiter='\t')
test_ratn_indices=np.load('test_ratn_indices.npy') #we may not need this!
test_ratn_values=np.load('test_ratn_values.npy')
item_sims=np.load('item_sims_original.npy')
no_users=2113
no_items=10197
for i in range(1,no_items):#row
    for j in range (0,i): #column
        item_sims[i,j]=item_sims[j,i] #we obtain symmetric item similarity matrix by this way.

k=30
preds=np.zeros(len(test_ratn_values))
for i in range(0,len(test_ratn_values)):
    print(i)
    rat_row=test_ratings.loc[i]
    usr_ID=int(rat_row["userID"])
    dfu = train_ratings[(train_ratings['userID']==usr_ID)]
    usr_ratings=dfu['rating'].to_numpy() #?
    mov_ID=int(rat_row["movieID"])
    mvin=np.where(mis==mov_ID) #mvin: target item mis-based index
    sorted_index_array = np.argsort(item_sims[mvin,:])
    desc_indices=np.flip(sorted_index_array) #take reverse of array. Note that mis-based index!
    nume=0
    deno=0
    usratd= dfu['movieID'].to_numpy()#list of user rated movies

    dic=[]
    for ix in range(0,k):
        dic.append(desc_indices[0,0,ix]) #dic holds k similar item indices
    rtneig=np.intersect1d(mis[dic],usratd)#lst_usr) 
    for j in range(0,len(rtneig)):
        j=0
        ji=np.where(mis==rtneig[j]) #finds index in specified user ratings
        rel_row=dfu[dfu['movieID']==int(rtneig[j])] #active user's related row including rating to be computed
        fnum=float(rel_row['rating'].iloc[0])*float(item_sims[mvin,ji])
        nume=nume+fnum #Data frame olup nan veriyorlar çok saçma
        deno=deno+abs(item_sims[mvin,ji])
    if deno!=0:
        pred=float(nume/deno)
    else:
        mn=dfu[['rating']].mean(axis=0)
        pred=float(mn.iloc[0])
    if pred>=0:
        if pred<=5 and pred>=0.5:
            preds[i]=pred
        elif pred<0.5:
            preds[i]=0.5
        else:
            preds[i]=5

#Computing NMAE
mae_sum=0
for i in range(0,len(test_ratn_values)):
    mae_sum=mae_sum+abs(test_ratn_values[i]-preds[i])
mae=mae_sum/len(test_ratn_values)
nmae=mae/(5-0.5) #Our rating range is [0.5, 5] with each step 0.5